import asyncio
import base64
import copy
import os
import pickle
import uuid
from pathlib import Path
from typing import Dict, List, Tuple

import streamlit as st
from langchain.storage import LocalFileStore
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, BaseMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph
from nio import AsyncClient, RoomMessageText, RoomMessageMedia, SyncError, RoomMessagesError, \
    AsyncClientConfig, RoomEncryptedMedia, DownloadError, Event
from nio.crypto import decrypt_attachment
from pydantic import BaseModel

from custom_cohere_embeddings import CustomCohereEmbeddings
from custom_milvus import CustomMilvus

if "event_loop" not in st.session_state:
    loop = asyncio.new_event_loop()
    st.session_state["event_loop"] = loop

loop = st.session_state["event_loop"]
asyncio.set_event_loop(loop)

store_path = Path("stores")
store_path.mkdir(exist_ok=True)

st.title("social-media-rag")

st.sidebar.header("Configurations")
cohere_api_key = st.sidebar.text_input(
    label="Cohere API Key",
    type="password",
    value=os.environ.get("COHERE_API_KEY")
)
google_api_key = st.sidebar.text_input(
    label="Google API Key",
    type="password",
    value=os.environ.get("GOOGLE_API_KEY")
)
zilliz_uri = st.sidebar.text_input(
    label="Zilliz URI",
    value=os.environ.get("ZILLIZ_URI")
)
zilliz_token = st.sidebar.text_input(
    label="Zilliz Token",
    type="password",
    value=os.environ.get("ZILLIZ_TOKEN")
)
matrix_user_id = st.sidebar.text_input(
    label="Matrix User ID",
    value=os.environ.get("MATRIX_USER_ID")
)
matrix_access_token = st.sidebar.text_input(
    label="Matrix Access Token",
    type="password",
    value=os.environ.get("MATRIX_ACCESS_TOKEN")
)
matrix_device_id = st.sidebar.text_input(
    label="Matrix Device ID",
    value=os.environ.get("MATRIX_DEVICE_ID")
)
matrix_keys_file = st.sidebar.file_uploader(
    label="Matrix Keys File",
    type="txt",
)
matrix_keys_passphrase = st.sidebar.text_input(
    label="Matrix Keys Passphrase",
    value=os.environ.get("MATRIX_KEYS_PASSPHRASE"),
    type="password",
)
matrix_room_id = st.sidebar.text_area(
    label="Matrix Room ID(s)",
    value=os.environ.get("MATRIX_ROOM_ID"),
    help="Enter one or more room IDs, separated by new lines."
)
matrix_fetch_limit = st.sidebar.number_input(
    label="Matrix Fetch Limit",
    value=200,
)
collection_name = st.sidebar.text_input(
    label="Collection Name",
    value="social_media_rag",
)
citation_limit = st.sidebar.number_input(
    label="Citation Limit",
    value=15,
)
embedder = CustomCohereEmbeddings(
    model="embed-v4.0",
    cohere_api_key=cohere_api_key,
)
generator_llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-pro",
    google_api_key=google_api_key,
)
vector_store = CustomMilvus(
    embedding_function=embedder,
    connection_args={"uri": zilliz_uri, "token": zilliz_token},
    collection_name=collection_name,
    enable_dynamic_field=True,
    index_params={"metric_type": "COSINE"},
    search_params={"metric_type": "COSINE"},
)
file_store = LocalFileStore(root_path=store_path / "file_store" / collection_name)


async def make_matrix_client():
    matrix_client = AsyncClient(
        homeserver="https://matrix.beeper.com",
        store_path=str(store_path),
        config=AsyncClientConfig(
            store_sync_tokens=True,
            encryption_enabled=True,
        )
    )
    matrix_client.restore_login(
        user_id=matrix_user_id,
        device_id=matrix_device_id,
        access_token=matrix_access_token,
    )

    keys_file_path = store_path / "keys.txt"
    if not keys_file_path.exists():
        if matrix_keys_file is None:
            st.error("Configure Matrix Keys File first.")
        else:
            keys_data = matrix_keys_file.getvalue()
            with open(keys_file_path, "wb") as f:
                f.write(keys_data)

    await matrix_client.import_keys(
        infile=str(keys_file_path),
        passphrase=matrix_keys_passphrase,
    )

    sync_response = await matrix_client.sync(full_state=True, timeout=30000)
    if isinstance(sync_response, SyncError):
        raise Exception(f"Sync failed: {sync_response}")

    return matrix_client


async def process_event(event: Event) -> List[Document]:
    matrix_client: AsyncClient = st.session_state["matrix_client"]
    if isinstance(event, RoomMessageText):
        document = Document(
            id=str(uuid.uuid5(uuid.NAMESPACE_OID, event.event_id)),
            page_content=event.body,
            metadata={
                "exclusion": {
                    "event_type": "text",
                    "event_source": event.source,
                }
            }
        )
        return [document]
    elif isinstance(event, RoomMessageMedia):
        mime_type = event.source["content"]["info"]["mimetype"]
        uri = await matrix_client.mxc_to_http(event.url)
        download_response = await matrix_client.download(event.url)
        if isinstance(download_response, DownloadError):
            raise Exception(f"Download failed: {download_response}")
        if mime_type.startswith("image/"):
            b64_uri = f"data:{mime_type};base64,{base64.b64encode(download_response.body).decode('utf-8')}"
            document = Document(
                id=str(uuid.uuid5(uuid.NAMESPACE_OID, event.event_id)),
                page_content=b64_uri,
                metadata={
                    "exclusion": {
                        "uri": uri,
                        "data": download_response.body,
                        "event_type": "media",
                        "event_source": event.source,
                    },
                }
            )
            return [document]
        else:
            st.warning(f"Unsupported mime type: {mime_type}")
            return []
    elif isinstance(event, RoomEncryptedMedia):
        mime_type = event.source["content"]["info"]["mimetype"]
        uri = await matrix_client.mxc_to_http(event.url)
        download_response = await matrix_client.download(event.url)
        if isinstance(download_response, DownloadError):
            raise Exception(f"Download failed: {download_response}")
        decrypted_data = decrypt_attachment(
            ciphertext=download_response.body,
            key=event.key["k"],
            hash=event.hashes["sha256"],
            iv=event.iv,
        )
        if mime_type.startswith("image/"):
            b64_uri = f"data:{mime_type};base64,{base64.b64encode(decrypted_data).decode('utf-8')}"
            document = Document(
                id=str(uuid.uuid5(uuid.NAMESPACE_OID, event.event_id)),
                page_content=b64_uri,
                metadata={
                    "exclusion": {
                        "uri": uri,
                        "data": decrypted_data,
                        "event_type": "encrypted_media",
                        "event_source": event.source,
                    },
                }
            )
            return [document]
        else:
            st.warning(f"Unsupported mime type: {mime_type}")
            return []
    else:
        st.warning(f"Unsupported event type: {event}")
        return []


class QNAState(BaseModel):
    query: str = None
    documents: List[Document] = None
    prompt: HumanMessage = None
    response: BaseMessage = None


async def retrieve(state: QNAState) -> QNAState:
    retrieved_documents: List[Tuple[Document, float]] = await vector_store.asimilarity_search_with_score(
        query=state.query,
        k=citation_limit
    )
    cached_documents: List[Document] = []
    for retrieved_document, score in retrieved_documents:
        document_id = retrieved_document.metadata["pk"]
        bytes_cached_document: bytes = (await file_store.amget([document_id]))[0]
        cached_document: Document = pickle.loads(bytes_cached_document)
        cached_document.metadata["retrieval_score"] = score
        cached_documents.append(cached_document)
    state.documents = cached_documents
    return state


async def get_citations(index, document) -> List[Dict]:
    event_type = document.metadata["exclusion"]["event_type"]
    if event_type == "text":
        return [
            {
                "type": "text",
                "text": f"""
                <citation_{index}>
                {document.page_content}
                </citation_{index}>
                """
            }
        ]
    elif event_type == "media":
        return [
            {
                "type": "text",
                "text": f"""
                <citation_{index}>
                """
            },
            {
                "type": "image_url",
                "image_url": document.metadata["exclusion"]["uri"],
            },
            {
                "type": "text",
                "text": f"""
                </citation_{index}>
                """
            },
        ]
    elif event_type == "encrypted_media":
        return [
            {
                "type": "text",
                "text": f"""
                <citation_{index}>
                """
            },
            {
                "type": "image_url",
                "image_url": document.page_content,
            },
            {
                "type": "text",
                "text": f"""
                </citation_{index}>
                """
            },
        ]
    else:
        raise Exception(f"Unsupported document type: {document}")


async def format_prompt(state: QNAState) -> QNAState:
    citations = []
    for index, document in enumerate(state.documents):
        citation = await get_citations(index + 1, document)
        citations.extend(citation)

    message_content = [
        {
            "type": "text",
            "text": f"""
            <instruction>
            Answer the following query using ONLY the provided citations.
            You MUST include the citations in your answer, i.e., [1, 2, 3, etc.].
            If you cannot find the answer in the citations, respond with "I don't have enough information to answer that query."
            Do not make up any information.
            </instruction>
            """
        },
        {
            "type": "text",
            "text": f"""
            <query>
            {query}
            </query>
            """
        },
        {
            "type": "text",
            "text": "<citations>"
        },
        *citations,
        {
            "type": "text",
            "text": "</citations>"
        },
    ]
    message = HumanMessage(content=message_content)
    state.prompt = message
    return state


async def generate(state: QNAState) -> QNAState:
    response: BaseMessage = await generator_llm.ainvoke([state.prompt])
    state.response = response
    return state


qna_graph = StateGraph(QNAState)
qna_graph.add_node("retrieve", retrieve)
qna_graph.add_node("format_prompt", format_prompt)
qna_graph.add_node("generate", generate)
qna_graph.add_edge("retrieve", "format_prompt")
qna_graph.add_edge("format_prompt", "generate")
qna_graph.set_entry_point("retrieve")
qna_graph.set_finish_point("generate")
qna_app = qna_graph.compile()

progress = st.empty()


async def fetch_messages(room_id, len_room_ids):
    matrix_client: AsyncClient = st.session_state["matrix_client"]
    with progress:
        st.session_state["progress_counter"] += 1
        st.progress(st.session_state["progress_counter"] / len_room_ids)
        st.text(f"Fetching {st.session_state["progress_counter"]}/{len_room_ids} room messages...")
    room_messages_response = await matrix_client.room_messages(room_id=room_id, limit=matrix_fetch_limit)
    if isinstance(room_messages_response, RoomMessagesError):
        raise Exception(f"Room messages fetch failed: {room_messages_response}")
    return room_messages_response


async def ingest_event(event: Event, len_events):
    with progress:
        st.session_state["progress_counter"] += 1
        st.progress(st.session_state["progress_counter"] / len_events)
        st.text(f"Ingesting {st.session_state["progress_counter"]}/{len_events} events...")

    documents: List[Document] = await process_event(event)

    document_ids = []
    bytes_documents = []
    vector_documents = []

    for document in documents:
        bytes_document = pickle.dumps(document)
        vector_document = copy.deepcopy(document)
        del vector_document.metadata["exclusion"]
        bytes_documents.append(bytes_document)
        vector_documents.append(vector_document)
        document_ids.append(document.id)

    if len(documents) > 0:
        await file_store.amset(list(zip(document_ids, bytes_documents)))
        if vector_store.col:
            await vector_store.adelete(ids=document_ids)
        await vector_store.aadd_documents(ids=document_ids, documents=vector_documents)


if st.sidebar.button("Sync", use_container_width=True):
    st.session_state["matrix_client"] = loop.run_until_complete(make_matrix_client())
    st.session_state["progress_counter"] = 0
    room_ids = matrix_room_id.split("\n")
    stored_ids = list(file_store.yield_keys())
    fetch_tasks = [fetch_messages(room_id, len(room_ids)) for room_id in room_ids]
    fetched_messages = loop.run_until_complete(asyncio.gather(*fetch_tasks))
    st.session_state["progress_counter"] = 0
    events = [
        event for fetched_message in fetched_messages for event in fetched_message.chunk
        if isinstance(event, (RoomMessageText, RoomMessageMedia, RoomEncryptedMedia))
           and str(uuid.uuid5(uuid.NAMESPACE_OID, event.event_id)) not in stored_ids
    ]
    ingest_tasks = [ingest_event(event, len(events)) for event in events]
    loop.run_until_complete(asyncio.gather(*ingest_tasks))
    st.session_state["progress_counter"] = 0
    st.success("Sync successfully!")

if st.sidebar.button("Reset", use_container_width=True):
    if vector_store.col:
        vector_store.col.drop()
    file_store.mdelete(keys=list(file_store.yield_keys()))

    for key in st.session_state.keys():
        del st.session_state[key]

    st.success("Reset successfully!")


@st.dialog(title="Citation Details", width="large")
def citation_details(document: Document):
    st.write(document.model_dump())


query = st.text_area("Ask a query:")
if st.button("Submit"):
    initial_state = QNAState(
        query=query,
    )
    final_state: QNAState = loop.run_until_complete(qna_app.ainvoke(initial_state))
    st.session_state["qna_result"] = final_state

if "qna_result" in st.session_state:
    st.markdown("**Response:**")
    st.write(st.session_state["qna_result"]["response"].content)
    st.markdown("**Citations:**")
    if len(st.session_state["qna_result"]["documents"]) > 0:
        for index, document in enumerate(st.session_state["qna_result"]["documents"]):
            event_type = document.metadata["exclusion"]["event_type"]
            if st.button(label=f"Citation {index + 1}: {document.metadata['retrieval_score']}"):
                citation_details(document)
            if event_type == "text":
                st.text(document.page_content)
            elif event_type in "media":
                uri = document.metadata["exclusion"]["uri"]
                st.image(uri)
            elif event_type in "encrypted_media":
                data = document.metadata["exclusion"]["data"]
                b64_data = base64.b64encode(data).decode("utf-8")
                mime_type = document.metadata["exclusion"]["event_source"]["content"]["info"]["mimetype"]
                uri = f"data:{mime_type};base64,{b64_data}"
                st.image(uri)
            else:
                raise Exception(f"Unsupported mime type: {document}")
    else:
        st.write("No citations found.")
