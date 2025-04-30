import asyncio
import base64
import hashlib
import io
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import streamlit as st
from graphlit import Graphlit
from graphlit_api import ContentFilter, ConversationInput, SpecificationInput, GoogleModels, ModelServiceTypes, \
    GoogleModelPropertiesInput, EntityReferenceInput, ConversationStrategyInput, RetrievalStrategyInput, \
    RetrievalStrategyTypes, ExtractionWorkflowStageInput, ExtractionWorkflowJobInput, EntityExtractionConnectorInput, \
    EntityExtractionServiceTypes, ModelImageExtractionPropertiesInput, SpecificationTypes, WorkflowInput, \
    Client
from nio import AsyncClient, RoomMessageText, RoomMessageMedia, SyncError, RoomMessagesError, \
    AsyncClientConfig, RoomEncryptedMedia, DownloadError
from nio.crypto import decrypt_attachment

loop = asyncio.new_event_loop()
asyncio.set_event_loop(loop)


async def make_graphlit_client():
    graphlit_client = Graphlit(
        organization_id=st.session_state["GRAPHLIT_ORGANIZATION_ID"],
        environment_id=st.session_state["GRAPHLIT_ENVIRONMENT_ID"],
        jwt_secret=st.session_state["GRAPHLIT_SECRET"],
    ).client
    return graphlit_client


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


async def ingest_text(event: RoomMessageText):
    graphlit_client: Client = st.session_state["graphlit_client"]
    data_byte = event.body.encode("utf-8")
    data_b64 = base64.b64encode(data_byte).decode("utf-8")
    content_hash = hashlib.sha256(data_byte).hexdigest()
    found_contents = await graphlit_client.query_contents(
        filter=ContentFilter(
            name=content_hash,
            limit=1,
        )
    )
    if len(found_contents.contents.results) == 0:
        return await graphlit_client.ingest_encoded_file(
            name=content_hash,
            data=data_b64,
            mime_type="text/plain",
            is_synchronous=True
        )
    return found_contents.contents.results[0]


async def ingest_media(event: RoomMessageMedia):
    matrix_client: AsyncClient = st.session_state["matrix_client"]
    graphlit_client: Client = st.session_state["graphlit_client"]
    content_hash = hashlib.sha256(event.url.encode("utf-8")).hexdigest()
    found_contents = await graphlit_client.query_contents(
        filter=ContentFilter(
            name=content_hash,
            limit=1,
        )
    )
    if len(found_contents.contents.results) == 0:
        uri = await matrix_client.mxc_to_http(event.url)
        create_specification_response = await graphlit_client.create_specification(
            specification=SpecificationInput(
                name=f"social-media-rag-extraction-specification",
                customInstructions="Create highly detailed descriptions.",
                type=SpecificationTypes.EXTRACTION,
                serviceType=ModelServiceTypes.GOOGLE,
                google=GoogleModelPropertiesInput(
                    model=GoogleModels.GEMINI_2_5_FLASH_PREVIEW
                ),
            )
        )
        create_workflow_response = await graphlit_client.create_workflow(
            workflow=WorkflowInput(
                name=f"social-media-rag-extraction-workflow",
                extraction=ExtractionWorkflowStageInput(
                    jobs=[
                        ExtractionWorkflowJobInput(
                            connector=EntityExtractionConnectorInput(
                                type=EntityExtractionServiceTypes.MODEL_IMAGE,
                                modelImage=ModelImageExtractionPropertiesInput(
                                    specification=EntityReferenceInput(
                                        id=create_specification_response.create_specification.id
                                    )
                                )
                            )
                        )
                    ]
                )
            )
        )

        if event.source["content"]["info"]["mimetype"].startswith("image/"):
            ingestion_workflow = EntityReferenceInput(id=create_workflow_response.create_workflow.id)
        else:
            ingestion_workflow = None

        return await graphlit_client.ingest_uri(
            name=content_hash,
            uri=uri,
            mime_type=event.source["content"]["info"]["mimetype"],
            is_synchronous=True,
            workflow=ingestion_workflow
        )
    return found_contents.contents.results[0]


async def ingest_encrypted_media(event: RoomEncryptedMedia):
    matrix_client: AsyncClient = st.session_state["matrix_client"]
    graphlit_client: Client = st.session_state["graphlit_client"]
    content_hash = hashlib.sha256(event.url.encode("utf-8")).hexdigest()
    found_contents = await graphlit_client.query_contents(
        filter=ContentFilter(
            name=content_hash,
            limit=1,
        )
    )
    if len(found_contents.contents.results) == 0:
        download_response = await matrix_client.download(event.url)
        if isinstance(download_response, DownloadError):
            raise Exception(f"Download failed: {download_response}")
        decrypted_data = decrypt_attachment(
            ciphertext=download_response.body,
            key=event.key["k"],
            hash=event.hashes["sha256"],
            iv=event.iv,
        )
        data_b64 = base64.b64encode(decrypted_data).decode("utf-8")

        create_specification_response = await graphlit_client.create_specification(
            specification=SpecificationInput(
                name=f"social-media-rag-extraction-specification",
                customInstructions="Create highly detailed descriptions.",
                type=SpecificationTypes.EXTRACTION,
                serviceType=ModelServiceTypes.GOOGLE,
                google=GoogleModelPropertiesInput(
                    model=GoogleModels.GEMINI_2_5_FLASH_PREVIEW
                ),
            )
        )
        create_workflow_response = await graphlit_client.create_workflow(
            workflow=WorkflowInput(
                name=f"social-media-rag-extraction-workflow",
                extraction=ExtractionWorkflowStageInput(
                    jobs=[
                        ExtractionWorkflowJobInput(
                            connector=EntityExtractionConnectorInput(
                                type=EntityExtractionServiceTypes.MODEL_IMAGE,
                                modelImage=ModelImageExtractionPropertiesInput(
                                    specification=EntityReferenceInput(
                                        id=create_specification_response.create_specification.id
                                    )
                                )
                            )
                        )
                    ]
                )
            )
        )

        if event.source["content"]["info"]["mimetype"].startswith("image/"):
            ingestion_workflow = EntityReferenceInput(id=create_workflow_response.create_workflow.id)
        else:
            ingestion_workflow = None

        return await graphlit_client.ingest_encoded_file(
            name=content_hash,
            data=data_b64,
            mime_type=event.mimetype,
            is_synchronous=True,
            workflow=ingestion_workflow
        )
    return found_contents.contents.results[0]


async def start_or_get_conversation():
    graphlit_client: Client = st.session_state["graphlit_client"]
    # for simplicity, we store one conversation per session.
    # clear cache if the user wants to start a new conversation.
    if "conversation_id" not in st.session_state:
        create_conversation = await graphlit_client.create_conversation(
            conversation=ConversationInput(
                name=f"social-media-rag-conversation-{datetime.now().isoformat()}",
            )
        )
        st.session_state["conversation_id"] = create_conversation.create_conversation.id
    return st.session_state["conversation_id"]


async def ask_rag(query: str):
    graphlit_client: Client = st.session_state["graphlit_client"]
    conversation_id = await start_or_get_conversation()
    prompt = f"""
        <instruction>
        Answer the following query using ONLY the citation that has been retrieved.
        If you cannot find the answer in the retrieved citation, you must respond with "I don't have enough information to answer that query.".
        Do not make up any information.
        <instruction/>
        <query>
        {query}
        <query/>
        """
    create_specification_response = await graphlit_client.create_specification(
        specification=SpecificationInput(
            name=f"social-media-rag-conversation-specification",
            type=SpecificationTypes.COMPLETION,
            serviceType=ModelServiceTypes.GOOGLE,
            google=GoogleModelPropertiesInput(
                model=GoogleModels.GEMINI_2_5_PRO_PREVIEW
            ),
            strategy=ConversationStrategyInput(
                embedCitations=True,
            ),
            retrievalStrategy=RetrievalStrategyInput(
                type=RetrievalStrategyTypes.CHUNK,
            ),
            numberSimilar=st.session_state["CITATION_LIMIT"]
        )
    )
    prompt_conversation_response = await graphlit_client.prompt_conversation(
        prompt=prompt,
        id=conversation_id,
        specification=EntityReferenceInput(
            id=create_specification_response.create_specification.id
        )
    )
    return prompt_conversation_response.prompt_conversation.message


# Streamlit UI
st.title("social-media-rag")

st.sidebar.header("Configurations")
st.sidebar.text_input(
    label="Graphlit Organization ID",
    key="GRAPHLIT_ORGANIZATION_ID",
    value=os.environ.get("GRAPHLIT_ORGANIZATION_ID"),
)
st.sidebar.text_input(
    label="Graphlit Environment ID",
    key="GRAPHLIT_ENVIRONMENT_ID",
    value=os.environ.get("GRAPHLIT_ENVIRONMENT_ID")
)
st.sidebar.text_input(
    label="Graphlit Secret",
    key="GRAPHLIT_SECRET",
    type="password",
    value=os.environ.get("GRAPHLIT_SECRET")
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
    value=15,
    min_value=1
)

if st.sidebar.button("Sync Configurations"):
    if not all(st.session_state[k] for k in (
            "GRAPHLIT_ORGANIZATION_ID",
            "GRAPHLIT_ENVIRONMENT_ID",
            "GRAPHLIT_SECRET",
            "MATRIX_USER_ID",
            "MATRIX_ACCESS_TOKEN",
            "MATRIX_DEVICE_ID",
            "MATRIX_KEYS_FILE",
            "MATRIX_KEYS_PASSPHRASE",
            "MATRIX_ROOM_ID",
            "CITATION_LIMIT",
    )):
        st.error("Fill in all Graphlit and Matrix details first.")
    else:
        st.session_state["matrix_client"] = loop.run_until_complete(make_matrix_client())
        st.session_state["graphlit_client"] = loop.run_until_complete(make_graphlit_client())
        if "rag_response" in st.session_state:
            del st.session_state["rag_response"]
        progress_bar = st.progress(0)
        progress_text = st.text("Initializing...")
        fetch_message_counter = 0
        process_event_counter = 0
        room_ids = st.session_state["MATRIX_ROOM_ID"].split("\n")


        async def fetch_messages(room_id):
            global fetch_message_counter, progress_bar, progress_text
            matrix_client: AsyncClient = st.session_state["matrix_client"]
            fetch_message_counter += 1
            progress_bar.progress(fetch_message_counter / len(room_ids))
            progress_text.text(f"Fetching {fetch_message_counter}/{len(room_ids)} room messages...")

            room_messages_response = await matrix_client.room_messages(
                room_id=room_id,
                limit=sys.maxsize,
            )
            if isinstance(room_messages_response, RoomMessagesError):
                raise Exception(f"Room messages fetch failed: {room_messages_response}")

            return room_messages_response


        fetch_messages_tasks = [fetch_messages(room_id) for room_id in room_ids]
        fetched_messages = loop.run_until_complete(asyncio.gather(*fetch_messages_tasks))
        events = []
        for fetched_message in fetched_messages:
            for event in fetched_message.chunk:
                if any([
                    isinstance(event, RoomMessageText),
                    isinstance(event, RoomMessageMedia),
                    isinstance(event, RoomEncryptedMedia),
                ]):
                    events.append(event)


        async def process_event(event):
            global process_event_counter, progress_bar, progress_text
            process_event_counter += 1
            progress_bar.progress(process_event_counter / len(events))
            progress_text.text(f"Processing {process_event_counter}/{len(events)} events...")

            if isinstance(event, RoomMessageText):
                return await ingest_text(event)
            elif isinstance(event, RoomMessageMedia):
                return await ingest_media(event)
            elif isinstance(event, RoomEncryptedMedia):
                return await ingest_encrypted_media(event)
            else:
                raise Exception(f"Unknown event type: {event}")


        process_event_tasks = [process_event(event) for event in events]
        process_event_results = loop.run_until_complete(asyncio.gather(*process_event_tasks))
        st.success("Sync completed successfully!")

if st.sidebar.button("Reset"):
    graphlit_client: Client = loop.run_until_complete(make_graphlit_client())
    loop.run_until_complete(graphlit_client.delete_all_contents())
    loop.run_until_complete(graphlit_client.delete_all_specifications())
    loop.run_until_complete(graphlit_client.delete_all_workflows())
    loop.run_until_complete(graphlit_client.delete_all_conversations())

    for key in st.session_state.keys():
        del st.session_state[key]

query = st.text_area("Ask a query:")
if st.button("Submit"):
    if not query:
        st.error("Enter a prompt.")
    else:
        rag_response = loop.run_until_complete(ask_rag(query))
        st.session_state["rag_response"] = rag_response


@st.dialog(title="Citation Details", width="large")
def citation_details(citation):
    citation_dict = json.loads(json.dumps(citation, default=lambda o: o.__dict__))
    st.write(citation_dict)


if "rag_response" in st.session_state:
    st.markdown("**Response:**")
    st.write(st.session_state["rag_response"].message)
    st.markdown("**Citations:**")
    if st.session_state["rag_response"].citations:
        for index, citation in enumerate(st.session_state["rag_response"].citations):
            if st.button(label=f"Citation {index + 1}"):
                citation_details(citation)
            if citation.content.mime_type.startswith("text/"):
                st.text(citation.text)
            elif citation.content.mime_type.startswith("image/"):
                st.image(citation.content.master_uri)
            elif citation.content.mime_type.startswith("application/"):
                st.link_button(
                    label=f"{citation.content.file_name}",
                    url=citation.content.master_uri,
                )
                st.write(citation.text)
            else:
                raise Exception(f"Unknown content type: {citation.content.mime_type}")
    else:
        st.write("No citation available.")
