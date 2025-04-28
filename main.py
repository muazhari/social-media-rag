import asyncio
import base64
import hashlib
import io
import os
import sys
from datetime import datetime
from pathlib import Path

import streamlit as st
from graphlit import Graphlit
from graphlit_api import ContentFilter, ConversationInput, SpecificationInput, GoogleModels, ModelServiceTypes, \
    GoogleModelPropertiesInput, EntityReferenceInput, ConversationStrategyInput, RetrievalStrategyInput, \
    RetrievalStrategyTypes
from nio import AsyncClient, RoomMessageText, RoomMessageMedia, SyncError, RoomMessagesError, \
    AsyncClientConfig, RoomEncryptedMedia


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
        user_id=st.session_state["MATRIX_USER"],
        device_id=st.session_state["MATRIX_DEVICE_ID"],
        access_token=st.session_state["MATRIX_TOKEN"],
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


async def retrieve_messages(room_ids: [str]):
    matrix_client = await make_matrix_client()
    room_messages_responses = []
    for room_id in room_ids:
        room_messages_response = await matrix_client.room_messages(
            room_id=room_id,
            limit=sys.maxsize,
        )
        if isinstance(room_messages_response, RoomMessagesError):
            raise Exception(f"Room messages fetch failed: {room_messages_response}")
        room_messages_responses.append(room_messages_response)

    messages = []
    for room_messages_response in room_messages_responses:
        for event in room_messages_response.chunk:
            print(type(event))
            print(event.__dict__)

            if isinstance(event, RoomMessageText):
                messages.append({
                    "type": "text",
                    "text": event.body,
                    "sender": event.sender,
                    "timestamp": event.server_timestamp,
                })
            elif isinstance(event, RoomMessageMedia):
                uri = await matrix_client.mxc_to_http(event.url)
                messages.append({
                    "type": "media",
                    "mxc": event.url,
                    "uri": uri,
                    "sender": event.sender,
                    "timestamp": event.server_timestamp,
                })
            elif isinstance(event, RoomEncryptedMedia):
                st.toast("Encrypted media message is not supported, skipping...")
    await matrix_client.close()
    return messages


async def ingest_text(message: dict):
    graphlit_client = await make_graphlit_client()
    data_byte = message["text"].encode("utf-8")
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


async def ingest_media(message: dict):
    graphlit_client = await make_graphlit_client()
    content_hash = hashlib.sha256(message["mxc"].encode("utf-8")).hexdigest()
    found_contents = await graphlit_client.query_contents(
        filter=ContentFilter(
            name=content_hash,
            limit=1,
        )
    )
    if len(found_contents.contents.results) == 0:
        return await graphlit_client.ingest_uri(
            name=content_hash,
            uri=message["uri"],
            is_synchronous=True
        )
    return found_contents.contents.results[0]


async def start_or_get_conversation():
    graphlit_client = await make_graphlit_client()
    # for simplicity, we store one conversation per session.
    # clear cache if the user wants to start a new conversation
    if "conversation_id" not in st.session_state:
        create_conversation = await graphlit_client.create_conversation(
            conversation=ConversationInput(
                name=f"social-media-rag-conversation-{datetime.now().isoformat()}",
            )
        )
        st.session_state["conversation_id"] = create_conversation.create_conversation.id
    return st.session_state["conversation_id"]


async def ask_rag(query: str):
    graphlit_client = await make_graphlit_client()
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
            name=f"social-media-rag-specification",
            serviceType=ModelServiceTypes.GOOGLE,
            google=GoogleModelPropertiesInput(
                model=GoogleModels.GEMINI_2_5_PRO_PREVIEW
            ),
            strategy=ConversationStrategyInput(
                embedCitations=True,
            ),
            retrievalStrategy=RetrievalStrategyInput(
                type=RetrievalStrategyTypes.CHUNK,
                contentLimit=st.session_state["GRAPHLIT_CITATION_LIMIT"],
            )
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


# --- Streamlit UI -----------------------------------------
st.title("social-media-rag")

st.sidebar.header("Credentials")
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
    label="Matrix User",
    key="MATRIX_USER",
    value=os.environ.get("MATRIX_USER")
)
st.sidebar.text_input(
    label="Matrix Token",
    key="MATRIX_TOKEN",
    type="password",
    value=os.environ.get("MATRIX_TOKEN")
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
    label="Graphlit Citation Limit",
    key="GRAPHLIT_CITATION_LIMIT",
    value=100,
    min_value=1,
    help="The maximum number of citations can be retrieved."
)

if st.sidebar.button("Sync"):
    if not all(st.session_state[k] for k in (
            "GRAPHLIT_ORGANIZATION_ID",
            "GRAPHLIT_ENVIRONMENT_ID",
            "GRAPHLIT_SECRET",
            "MATRIX_USER",
            "MATRIX_TOKEN",
            "MATRIX_DEVICE_ID",
            "MATRIX_KEYS_FILE",
            "MATRIX_KEYS_PASSPHRASE",
            "MATRIX_ROOM_ID",
            "GRAPHLIT_CITATION_LIMIT",
    )):
        st.error("Fill in all Graphlit and Matrix details first.")
    else:
        with st.spinner("Fetching messages..."):
            room_ids = st.session_state["MATRIX_ROOM_ID"].split("\n")
            messages = asyncio.run(retrieve_messages(room_ids=room_ids))

        progress_bar = st.progress(0)
        progress_text = st.text("Ingesting messages...")
        counter = 0


        async def process_message(message):
            global counter
            if message["type"] == "text":
                await ingest_text(message)
            elif message["type"] == "media":
                await ingest_media(message)
            else:
                raise Exception(f"Unknown message type: {message['type']}")

            counter += 1
            progress_bar.progress(counter / len(messages))
            progress_text.text(f"Ingesting {counter}/{len(messages)} messages...")


        tasks = [process_message(message) for message in messages]
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(asyncio.gather(*tasks))

        st.success("Sync completed successfully!")

query = st.text_area("Ask a query:")
if st.button("Submit"):
    if not query:
        st.error("Enter a prompt.")
    else:
        rag_response = asyncio.run(ask_rag(query))
        st.markdown("**Response:**")
        st.write(rag_response.message)
        st.markdown("**Citations:**")
        if rag_response.citations:
            for index, citation in enumerate(rag_response.citations):
                st.page_link(label=f"Citation {index + 1}", page=citation.content.master_uri)
                if citation.content.mime_type.startswith("text/"):
                    st.text(citation.text)
                elif citation.content.mime_type.startswith("image/"):
                    st.image(citation.content.uri)
                else:
                    raise Exception(f"Unknown content type: {citation.content.mime_type}")
