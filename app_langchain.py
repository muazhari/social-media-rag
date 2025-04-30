import asyncio
import base64
import hashlib
import io
import os
import sys
from pathlib import Path

import streamlit as st
from langchain_cohere import CohereEmbeddings
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, BaseMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_milvus.vectorstores.milvus import Milvus
from langgraph.graph import Graph
from langgraph.graph.graph import CompiledGraph
from nio import AsyncClient, RoomMessageText, RoomMessageMedia, SyncError, RoomMessagesError, \
    AsyncClientConfig, RoomEncryptedMedia, DownloadError, Event
from nio.crypto import decrypt_attachment


async def make_matrix_client():
    store_dir = Path("./encryption_keys")
    store_dir.mkdir(exist_ok=True)
    matrix_client = AsyncClient(
        homeserver="https://matrix.beeper.com",
        store_path=str(store_dir),
        config=AsyncClientConfig(
            store_sync_tokens=True,
            encryption_enabled=True,
        )
    )
    matrix_client.restore_login(
        user_id=st.session_state["MATRIX_USER_ID"],
        device_id=st.session_state["MATRIX_DEVICE_ID"],
        access_token=st.session_state["MATRIX_ACCESS_TOKEN"],
    )

    keys_file_path = str(store_dir / "keys.txt")
    current_keys_data = st.session_state["MATRIX_KEYS_FILE"].getvalue()
    current_keys_hash = hashlib.sha256(current_keys_data).hexdigest()

    last_keys_file = io.FileIO(keys_file_path)
    last_keys_data = last_keys_file.read()
    last_keys_hash = hashlib.sha256(last_keys_data).hexdigest()

    if last_keys_hash != current_keys_hash:
        with open(keys_file_path, "wb") as f:
            f.write(current_keys_data)

        await matrix_client.import_keys(
            infile=keys_file_path,
            passphrase=st.session_state["MATRIX_KEYS_PASSPHRASE"],
        )

    sync_response = await matrix_client.sync(full_state=True, timeout=30000)
    if isinstance(sync_response, SyncError):
        raise Exception(f"Sync failed: {sync_response}")

    return matrix_client


async def describe_image(uri: str):
    llm: ChatGoogleGenerativeAI = st.session_state["ingestor_llm"]
    content = [
        {
            "type": "text",
            "text": f"""
            <instruction>
            Create highly detailed descriptions of the image.
            </instruction>
            """
        },
        {
            "type": "image_url",
            "image_url": uri
        },
    ]
    message: HumanMessage = HumanMessage(content=content)
    response: BaseMessage = llm.invoke([message])
    return response.content


async def process_event(event: Event) -> [Document]:
    print(event.source)
    matrix_client: AsyncClient = st.session_state["matrix_client"]
    if isinstance(event, RoomMessageText):
        id = hashlib.sha256(event.body.encode()).hexdigest()
        document = Document(
            id=id,
            page_content=event.body,
            metadata={
                "event_type": "text",
                "event_source": event.source,
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
            page_content = await describe_image(uri)
            id = hashlib.sha256(download_response.body).hexdigest()
            document = Document(
                id=id,
                page_content=page_content,
                metadata={
                    "event_type": "media",
                    "event_source": event.source,
                }
            )
            return [document]
        else:
            st.warning(f"Unsupported mime type: {mime_type}")
            return []
    elif isinstance(event, RoomEncryptedMedia):
        mime_type = event.source["content"]["info"]["mimetype"]
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
            page_content = await describe_image(b64_uri)
            id = hashlib.sha256(decrypted_data).hexdigest()
            document = Document(
                id=id,
                page_content=page_content,
                metadata={
                    "event_type": "encrypted_media",
                    "event_source": event.source,
                }
            )
            return [document]
        else:
            st.warning(f"Unsupported mime type: {mime_type}")
            return []
    else:
        raise Exception(f"Unsupported event type: {event}")


async def retrieve(state):
    query: str = state["query"]
    vector_store: Milvus = st.session_state["vector_store"]
    documents: [Document] = vector_store.similarity_search(query, k=st.session_state["CITATION_LIMIT"])
    state["documents"] = documents
    return state


async def get_citations(index, document) -> [dict]:
    matrix_client: AsyncClient = st.session_state["matrix_client"]
    event_type = document.metadata["event_type"]
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
        url = await matrix_client.mxc_to_http(document.metadata["event_source"]["url"])
        return [
            {
                "type": "text",
                "text": f"""
                <citation_{index}>
                """
            },
            {
                "type": "image_url",
                "image_url": url,
            },
            {
                "type": "text",
                "text": f"""
                </citation_{index}>
                """
            },
        ]
    elif event_type == "encrypted_media":
        mime_type = document.metadata["event_source"]["content"]["info"]["mimetype"]
        download_response = await matrix_client.download(document.metadata["event_source"]["content"]["file"]["url"])
        if isinstance(download_response, DownloadError):
            raise Exception(f"Download failed: {download_response}")
        decrypted_data = decrypt_attachment(
            ciphertext=download_response.body,
            key=document.metadata["event_source"]["content"]["file"]["key"]["k"],
            hash=document.metadata["event_source"]["content"]["file"]["hashes"]["sha256"],
            iv=document.metadata["event_source"]["content"]["file"]["iv"],
        )
        b64_data = base64.b64encode(decrypted_data).decode("utf-8")
        url = f"data:{mime_type};base64,{b64_data}"
        return [
            {
                "type": "text",
                "text": f"""
                <citation_{index}>
                """
            },
            {
                "type": "image_url",
                "image_url": url,
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


async def format_prompt(state):
    query: str = state["query"]
    documents: [Document] = state["documents"]

    citations = []
    for index, document in enumerate(documents):
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
    state["prompt"] = message
    return state


async def generate(state):
    prompt: HumanMessage = state["prompt"]
    llm: ChatGoogleGenerativeAI = st.session_state["generator_llm"]
    response: BaseMessage = llm.invoke([prompt])
    state["response"] = response.content
    return state


st.title("social-media-rag")

st.sidebar.header("Configurations")
st.sidebar.text_input(
    label="Cohere API Key",
    key="COHERE_API_KEY",
    type="password",
    value=os.environ.get("COHERE_API_KEY")
)
st.sidebar.text_input(
    label="Google API Key",
    key="GOOGLE_API_KEY",
    type="password",
    value=os.environ.get("GOOGLE_API_KEY")
)
st.sidebar.text_input(
    label="Zilliz URI",
    key="ZILLIZ_URI",
    value=os.environ.get("ZILLIZ_URI")
)
st.sidebar.text_input(
    label="Zilliz Token",
    key="ZILLIZ_TOKEN",
    type="password",
    value=os.environ.get("ZILLIZ_TOKEN")
)
st.sidebar.text_input(
    label="Matrix User ID",
    key="MATRIX_USER_ID",
    value=os.environ.get("MATRIX_USER_ID")
)
st.sidebar.text_input(
    label="Matrix Access Token",
    key="MATRIX_ACCESS_TOKEN",
    type="password",
    value=os.environ.get("MATRIX_ACCESS_TOKEN")
)
st.sidebar.text_input(
    label="Matrix Device ID",
    key="MATRIX_DEVICE_ID",
    value=os.environ.get("MATRIX_DEVICE_ID")
)
st.sidebar.file_uploader(
    label="Matrix Keys File",
    type="txt",
    key="MATRIX_KEYS_FILE"
)
st.sidebar.text_input(
    label="Matrix Keys Passphrase",
    key="MATRIX_KEYS_PASSPHRASE",
    value=os.environ.get("MATRIX_KEYS_PASSPHRASE"),
    type="password"
)
st.sidebar.text_area(
    label="Matrix Room ID(s)",
    key="MATRIX_ROOM_ID",
    value=os.environ.get("MATRIX_ROOM_ID"),
    help="Enter one or more room IDs, separated by new lines."
)
st.sidebar.number_input(
    label="Citation Limit",
    key="CITATION_LIMIT",
    value=100,
    min_value=1,
)

is_do_ingestion = st.sidebar.checkbox(
    label="Do Ingestion",
    value=True,
)

if st.sidebar.button("Sync", use_container_width=True):
    required_keys = [
        "COHERE_API_KEY", "GOOGLE_API_KEY", "ZILLIZ_URI", "ZILLIZ_TOKEN",
        "MATRIX_USER_ID", "MATRIX_ACCESS_TOKEN", "MATRIX_DEVICE_ID",
        "MATRIX_KEYS_FILE", "MATRIX_KEYS_PASSPHRASE", "MATRIX_ROOM_ID",
        "CITATION_LIMIT"
    ]
    if not all(st.session_state.get(k) for k in required_keys):
        st.error("Configure first.")
    else:
        matrix_client: AsyncClient = asyncio.get_event_loop().run_until_complete(make_matrix_client())
        st.session_state["matrix_client"] = matrix_client

        embeddings = CohereEmbeddings(
            model="embed-v4.0",
            cohere_api_key=st.session_state["COHERE_API_KEY"]
        )
        generator_llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-pro-exp-03-25",
            google_api_key=st.session_state["GOOGLE_API_KEY"]
        )
        ingestor_llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash-preview-04-17",
            google_api_key=st.session_state["GOOGLE_API_KEY"]
        )
        vector_store = Milvus(
            embedding_function=embeddings,
            connection_args={"uri": st.session_state["ZILLIZ_URI"], "token": st.session_state["ZILLIZ_TOKEN"]},
            collection_name="social_media_rag",
            enable_dynamic_field=True,
        )
        st.session_state["embeddings"] = embeddings
        st.session_state["generator_llm"] = generator_llm
        st.session_state["ingestor_llm"] = ingestor_llm
        st.session_state["vector_store"] = vector_store

        graph = Graph()
        graph.add_node("retrieve", retrieve)
        graph.add_node("format_prompt", format_prompt)
        graph.add_node("generate", generate)
        graph.add_edge("retrieve", "format_prompt")
        graph.add_edge("format_prompt", "generate")
        graph.set_entry_point("retrieve")
        graph.set_finish_point("generate")
        qna_graph = graph.compile()
        st.session_state["qna_graph"] = qna_graph

        progress_bar = st.progress(0)
        progress_text = st.text("Initializing...")
        fetch_message_counter = 0
        ingest_event_counter = 0
        room_ids = st.session_state["MATRIX_ROOM_ID"].split("\n")


        async def fetch_messages(room_id):
            global fetch_message_counter
            matrix_client: AsyncClient = st.session_state["matrix_client"]
            fetch_message_counter += 1
            progress_bar.progress(fetch_message_counter / len(room_ids))
            progress_text.text(f"Fetching {fetch_message_counter}/{len(room_ids)} room messages...")
            room_messages_response = await matrix_client.room_messages(room_id=room_id, limit=sys.maxsize)
            if isinstance(room_messages_response, RoomMessagesError):
                raise Exception(f"Room messages fetch failed: {room_messages_response}")
            return room_messages_response


        async def ingest_event(event: Event):
            global ingest_event_counter
            ingest_event_counter += 1
            progress_bar.progress(ingest_event_counter / len(events))
            progress_text.text(f"Ingesting {ingest_event_counter}/{len(events)} events...")
            documents = await process_event(event)
            if len(documents) > 0:
                ids = [document.id for document in documents]
                await vector_store.aadd_documents(ids=ids, documents=documents)


        if is_do_ingestion:
            fetch_tasks = [fetch_messages(room_id) for room_id in room_ids]
            fetched_messages = asyncio.get_event_loop().run_until_complete(asyncio.gather(*fetch_tasks))
            events = [
                event for fetched_message in fetched_messages for event in fetched_message.chunk
                if isinstance(event, (RoomMessageText, RoomMessageMedia, RoomEncryptedMedia))
            ]
            ingest_tasks = [ingest_event(event) for event in events]
            asyncio.get_event_loop().run_until_complete(asyncio.gather(*ingest_tasks))

        st.success("Sync successfully!")

if st.sidebar.button("Reset", use_container_width=True):
    required_keys = ["vector_store"]
    if not all(st.session_state.get(k) for k in required_keys):
        st.error("Configure first.")
    else:
        milvus: Milvus = st.session_state["vector_store"]
        if milvus.col:
            milvus.col.drop()

        for key in list(st.session_state.keys()):
            del st.session_state[key]

        st.success("Reset successfully!")

query = st.text_area("Ask a query:")
if st.button("Submit"):
    required_keys = [
        "COHERE_API_KEY", "GOOGLE_API_KEY", "ZILLIZ_URI", "ZILLIZ_TOKEN",
        "MATRIX_USER_ID", "MATRIX_ACCESS_TOKEN", "MATRIX_DEVICE_ID",
        "MATRIX_KEYS_FILE", "MATRIX_KEYS_PASSPHRASE", "MATRIX_ROOM_ID",
        "CITATION_LIMIT", "vector_store", "qna_graph"
    ]
    if not all(st.session_state.get(k) for k in required_keys):
        st.error("Configure first.")
    else:
        matrix_client: AsyncClient = asyncio.get_event_loop().run_until_complete(make_matrix_client())
        st.session_state["matrix_client"] = matrix_client

        initial_state = {"query": query}
        qna_graph: CompiledGraph = st.session_state["qna_graph"]
        final_state = asyncio.get_event_loop().run_until_complete(qna_graph.ainvoke(initial_state))
        st.session_state["rag_result"] = final_state


@st.dialog(title="Citation Details", width="large")
def citation_details(document: Document):
    st.write(document.model_dump())


if "rag_result" in st.session_state:
    matrix_client: AsyncClient = asyncio.get_event_loop().run_until_complete(make_matrix_client())
    st.session_state["matrix_client"] = matrix_client
    st.markdown("**Response:**")
    st.write(st.session_state["rag_result"]["response"])
    st.markdown("**Citations:**")
    if st.session_state["rag_result"]["documents"]:
        for index, document in enumerate(st.session_state["rag_result"]["documents"]):
            event_type = document.metadata["event_type"]
            if st.button(label=f"Citation {index + 1}"):
                citation_details(document)
            if event_type == "text":
                st.text(document.page_content)
            elif event_type in "media":
                uri = matrix_client.mxc_to_http(document.metadata["event_source"]["file"]["url"])
                st.image(uri)
            elif event_type in "encrypted_media":
                mxc = document.metadata["event_source"]["content"]["file"]["url"]
                download_response = asyncio.get_event_loop().run_until_complete(matrix_client.download(mxc))
                if isinstance(download_response, DownloadError):
                    raise Exception(f"Download failed: {download_response}")
                decrypted_data = decrypt_attachment(
                    ciphertext=download_response.body,
                    key=document.metadata["event_source"]["content"]["file"]["key"]["k"],
                    hash=document.metadata["event_source"]["content"]["file"]["hashes"]["sha256"],
                    iv=document.metadata["event_source"]["content"]["file"]["iv"],
                )
                b64_data = base64.b64encode(decrypted_data).decode("utf-8")
                mime_type = document.metadata["event_source"]["content"]["info"]["mimetype"]
                uri = f"data:{mime_type};base64,{b64_data}"
                st.image(uri)
            else:
                raise Exception(f"Unsupported mime type: {document}")
    else:
        st.write("No citations available.")
