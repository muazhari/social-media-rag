import asyncio
import base64
import hashlib
import os
from pathlib import Path

import streamlit as st
from graphlit import Graphlit
from nio import AsyncClient, RoomMessageText, RoomMessageMedia, SyncError, LoginError, RoomMessagesError, \
    AsyncClientConfig, MegolmEvent


# --- Async Matrix fetcher -----------------------------------
async def retrieve_messages(user, token, device_id, room_id):
    store_dir = Path("./encryption_keys")
    store_dir.mkdir(exist_ok=True)

    client = AsyncClient(
        homeserver="https://matrix.beeper.com",
        user=user,
        device_id=device_id,
        store_path=str(store_dir),
        config=AsyncClientConfig(
            store_sync_tokens=True,
            encryption_enabled=True,
        )
    )
    client.restore_login(
        user_id=user,
        device_id=device_id,
        access_token=token,
    )

    sync_response = await client.sync(timeout=30000)
    if isinstance(sync_response, SyncError):
        raise Exception(f"Sync failed: {sync_response}")
    room_messages_response = await client.room_messages(room_id, limit=100)
    if isinstance(room_messages_response, RoomMessagesError):
        raise Exception(f"Room messages fetch failed: {room_messages_response}")
    messages = []
    for event in room_messages_response.chunk:
        if isinstance(event, RoomMessageText):
            messages.append({
                "type": "text",
                "text": event.body,
                "sender": event.sender,
                "timestamp": event.server_timestamp,
            })
        elif isinstance(event, RoomMessageMedia):
            uri = await client.mxc_to_http(event.url)
            messages.append({
                "type": "media",
                "uri": uri,
                "sender": event.sender,
                "timestamp": event.server_timestamp,
            })
        elif isinstance(event, MegolmEvent):
            decrypted = client.decrypt_event(event)
            if decrypted.decrypted:
                print(decrypted)
            else:
                raise Exception(f"Unsupported message type: {type(decrypted)}")
    await client.close()
    return messages


# --- Graphlit helpers -------------------------------------
def make_client():
    return Graphlit(
        organization_id=st.session_state["GRAPHLIT_ORGANIZATION_ID"],
        environment_id=st.session_state["GRAPHLIT_ENVIRONMENT_ID"],
        jwt_secret=st.session_state["GRAPHLIT_SECRET"],
    ).client  # .client is the GraphQL client


async def ingest_text(text: str):
    # wrap text as bytes
    data_byte = text.encode("utf-8")
    data_b64 = base64.b64encode(data_byte).decode("utf-8")
    content_hash = hashlib.sha256(data_byte).hexdigest()
    found_contents = await make_client().query_contents(
        filter={
            "name": content_hash,
            "limit": 1,
        }
    )
    if len(found_contents.contents.results) == 0:
        return await make_client().ingest_encoded_file(
            name=content_hash,
            data=data_b64,
            mime_type="text/plain",
            is_synchronous=True
        )
    return found_contents[0]


async def ingest_media(uri: str):
    content_hash = hashlib.sha256(uri.encode("utf-8")).hexdigest()
    found_contents = await make_client().query_contents(
        filter={
            "name": content_hash,
            "limit": 1,
        }
    )
    if len(found_contents.contents.results) == 0:
        return await make_client().ingest_uri(
            name=content_hash,
            uri=uri,
            is_synchronous=True
        )
    return found_contents[0]


async def start_or_get_conversation():
    # for simplicity, we store one conversation per session
    if "conversation_id" not in st.session_state:
        # Create a conversation object with minimal required fields
        conversation = {"name": "Matrix Chat Session"}
        create_conversation = await make_client().create_conversation(conversation)
        st.session_state["conversation_id"] = create_conversation.create_conversation.id
    return st.session_state["conversation_id"]


async def ask_rag(query: str):
    conversation_id = await start_or_get_conversation()
    prompt = f"""
        Answer the following query using ONLY the citation that has been retrieved.
        If you cannot find the answer in the retrieved citation, you must respond with "I don't have enough information to answer that query."
        Do not make up any information other than from the retrieved citation.

        Query: {query}
        """
    create_specification_response = await make_client().create_specification(
        specification={
            "name": "Matrix Chat Specification",
            "serviceType": "OPEN_AI",
            "openAI": {
                "model": "GPT4O_128K"
            },
            "strategy": {
                "embedCitations": True,
            }
        }
    )
    prompt_conversation_response = await make_client().prompt_conversation(
        prompt=prompt,
        id=conversation_id,
        specification={
            "id": create_specification_response.create_specification.id,
        }
    )
    return prompt_conversation_response.prompt_conversation.message


# --- Streamlit UI -----------------------------------------
st.title("Matrix → Graphlit RAG")

st.sidebar.header("Credentials")
st.sidebar.text_input("Graphlit Organization ID", key="GRAPHLIT_ORGANIZATION_ID",
                      value=os.environ.get("GRAPHLIT_ORGANIZATION_ID"))
st.sidebar.text_input("Graphlit Environment ID", key="GRAPHLIT_ENVIRONMENT_ID",
                      value=os.environ.get("GRAPHLIT_ENVIRONMENT_ID"))
st.sidebar.text_input("Graphlit Secret", key="GRAPHLIT_SECRET", type="password",
                      value=os.environ.get("GRAPHLIT_SECRET"))
st.sidebar.text_input("Matrix User", key="MATRIX_USER", value=os.environ.get("MATRIX_USER"))
st.sidebar.text_input("Matrix Token", key="MATRIX_TOKEN", type="password",
                      value=os.environ.get("MATRIX_TOKEN"))
st.sidebar.text_input("Matrix Device ID", key="MATRIX_DEVICE_ID", value=os.environ.get("MATRIX_DEVICE_ID"))
st.sidebar.text_input("Matrix Room ID", key="MATRIX_ROOM_ID", value=os.environ.get("MATRIX_ROOM_ID"))

if st.sidebar.button("Sync from Matrix"):
    if not all(st.session_state[k] for k in (
            "GRAPHLIT_ORGANIZATION_ID", "GRAPHLIT_ENVIRONMENT_ID", "GRAPHLIT_SECRET", "MATRIX_TOKEN",
            "MATRIX_ROOM_ID")):
        st.error("Fill in all credentials and Matrix details first.")
    else:
        with st.spinner("Fetching messages…"):
            messages = asyncio.run(
                retrieve_messages(
                    user=st.session_state["MATRIX_USER"],
                    token=st.session_state["MATRIX_TOKEN"],
                    device_id=st.session_state["MATRIX_DEVICE_ID"],
                    room_id=st.session_state["MATRIX_ROOM_ID"]
                )
            )
            for message in messages:
                if message["type"] == "text":
                    asyncio.run(ingest_text(message["text"]))
                elif message["type"] == "media":
                    asyncio.run(ingest_media(message["uri"]))
                else:
                    raise Exception(f"Unknown message type: {message['type']}")
        st.success("Synced to Graphlit!")

query = st.text_area("Ask a query:")
if st.button("Submit"):
    if not query:
        st.error("Enter a prompt.")
    else:
        rag_response = asyncio.run(ask_rag(query))
        st.write(rag_response.message)
        if rag_response.citations:
            for index, citation in enumerate(rag_response.citations):
                st.page_link(label=f"**Citation {index + 1}:**", page=citation.content.master_uri)
                if citation.content.mime_type.startswith("text/"):
                    st.markdown(citation.text)
                elif citation.content.mime_type.startswith("image/"):
                    st.image(citation.content.text_uri)
                else:
                    raise Exception(f"Unknown content type: {citation.content.mime_type}")
