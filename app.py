import asyncio
import base64
import io
import os
import uuid
from pathlib import Path

import nest_asyncio
import streamlit as st
from langchain_core.documents import Document

from internals.containers.app_container import AppContainer
from internals.models.config import AppConfig, SessionConfig

if "event_loop" not in st.session_state:
    st.session_state["event_loop"] = asyncio.new_event_loop()

loop = st.session_state["event_loop"]
asyncio.set_event_loop(loop)
nest_asyncio.apply()


async def main():
    st.title("Social Media RAG")

    # Sidebar: App-level configs
    st.sidebar.header("App Configurations")

    app_config_toggle = st.sidebar.toggle(
        "Use server environment variables.",
        key="app_config_toggle",
        value=True
    )

    if app_config_toggle:
        cohere_api_key = os.getenv("COHERE_API_KEY")
        zilliz_uri = os.getenv("ZILLIZ_URI")
        zilliz_token = os.getenv("ZILLIZ_TOKEN")
        zilliz_collection_name = os.getenv("ZILLIZ_COLLECTION_NAME")
        google_api_key = os.getenv("GOOGLE_API_KEY")
        db_url = os.getenv("DB_URL")
        s3_bucket = os.getenv("S3_BUCKET")
        s3_endpoint_url = os.getenv("S3_ENDPOINT_URL")
        s3_access_key_id = os.getenv("S3_ACCESS_KEY_ID")
        s3_secret_access_key = os.getenv("S3_SECRET_ACCESS_KEY")
        s3_region_name = os.getenv("S3_REGION_NAME")
    else:
        cohere_api_key = st.sidebar.text_input(
            "Cohere API Key",
            type="password",
            key="COHERE_API_KEY",
            value=st.session_state.get("COHERE_API_KEY")
        )
        zilliz_uri = st.sidebar.text_input(
            "Zilliz URI",
            key="ZILLIZ_URI",
            value=st.session_state.get("ZILLIZ_URI")
        )
        zilliz_token = st.sidebar.text_input(
            "Zilliz Token",
            type="password",
            key="ZILLIZ_TOKEN",
            value=st.session_state.get("ZILLIZ_TOKEN")
        )
        zilliz_collection_name = st.sidebar.text_input(
            "Zilliz Collection Name",
            key="ZILLIZ_COLLECTION_NAME",
            value=st.session_state.get("ZILLIZ_COLLECTION_NAME", "social_media_rag")
        )
        google_api_key = st.sidebar.text_input(
            "Google API Key", type="password",
            key="GOOGLE_API_KEY",
            value=st.session_state.get("GOOGLE_API_KEY")
        )
        db_url = st.sidebar.text_input(
            "Database URL",
            key="DB_URL",
            value=st.session_state.get("DB_URL")
        )
        s3_bucket = st.sidebar.text_input(
            "S3 Bucket",
            key="SUPABASE_S3_BUCKET",
            value=st.session_state.get("S3_BUCKET")
        )
        s3_region_name = st.sidebar.text_input(
            "S3 Region Name",
            key="S3_REGION_NAME",
            value=st.session_state.get("S3_REGION_NAME")
        )
        s3_endpoint_url = st.sidebar.text_input(
            "S3 Endpoint URL",
            key="S3_ENDPOINT_URL",
            value=st.session_state.get("S3_ENDPOINT_URL")
        )
        s3_access_key_id = st.sidebar.text_input(
            "S3 Access Key ID",
            key="S3_ACCESS_KEY_ID",
            value=st.session_state.get("S3_ACCESS_KEY_ID")
        )
        s3_secret_access_key = st.sidebar.text_input(
            "S3 Secret Access Key",
            type="password",
            key="S3_SECRET_ACCESS_KEY",
            value=st.session_state.get("S3_SECRET_ACCESS_KEY")
        )

    # Sidebar: Session-level configs
    st.sidebar.header("Session Configurations")

    session_config_toggle = st.sidebar.toggle(
        "Use server environment variables.",
        key="session_config_toggle",
        value=False
    )

    if session_config_toggle:
        session_id = os.getenv("SESSION_ID")
        matrix_user_id = os.getenv("MATRIX_USER_ID")
        matrix_access_token = os.getenv("MATRIX_ACCESS_TOKEN")
        matrix_device = os.getenv("MATRIX_DEVICE_ID")
        keys_file_path = Path(__file__).parent / "stores" / "keys.txt"
        if keys_file_path.exists():
            matrix_keys_file = io.FileIO(file=keys_file_path, mode='rb')
        else:
            matrix_keys_file = None
        matrix_keys_passphrase = os.getenv("MATRIX_KEYS_PASSPHRASE")
        matrix_room_ids = os.getenv("MATRIX_ROOM_IDS")
    else:
        session_id = st.sidebar.text_input(
            "Session ID",
            key="SESSION_ID",
            value=st.session_state.get("SESSION_ID", str(uuid.uuid4()))
        )
        matrix_user_id = st.sidebar.text_input(
            "Matrix User ID",
            key="MATRIX_USER_ID",
            value=st.session_state.get("MATRIX_USER_ID")
        )
        matrix_access_token = st.sidebar.text_input(
            "Matrix Access Token",
            type="password",
            key="MATRIX_ACCESS_TOKEN",
            value=st.session_state.get("MATRIX_ACCESS_TOKEN")
        )
        matrix_device = st.sidebar.text_input(
            "Matrix Device ID",
            key="MATRIX_DEVICE_ID",
            value=st.session_state.get("MATRIX_DEVICE_ID")
        )
        matrix_keys_file = st.sidebar.file_uploader(
            "Matrix Keys File",
            type="txt",
            key="MATRIX_KEYS_FILE",
        )
        matrix_keys_passphrase = st.sidebar.text_input(
            "Matrix Keys Passphrase",
            type="password",
            key="MATRIX_KEYS_PASSPHRASE",
            value=st.session_state.get("MATRIX_KEYS_PASSPHRASE")
        )
        matrix_room_ids = st.sidebar.text_area(
            "Matrix Room IDs (comma-separated)",
            key="MATRIX_ROOM_IDS",
            value=st.session_state.get("MATRIX_ROOM_IDS"),
        )

    fetch_limit = st.sidebar.number_input(
        "Fetch Limit",
        value=st.session_state.get("FETCH_LIMIT", 200),
        key="FETCH_LIMIT",
    )
    citation_limit = st.sidebar.number_input(
        "Citation Limit",
        value=st.session_state.get("CITATION_LIMIT", 5),
        key="CITATION_LIMIT"
    )

    config_map = {
        "Cohere API Key": cohere_api_key,
        "Zilliz URI": zilliz_uri,
        "Zilliz Token": zilliz_token,
        "Zilliz Collection Name": zilliz_collection_name,
        "Google API Key": google_api_key,
        "Database URL": db_url,
        "S3 Bucket": s3_bucket,
        "S3 Region Name": s3_region_name,
        "S3 Endpoint URL": s3_endpoint_url,
        "S3 Access Key ID": s3_access_key_id,
        "S3 Secret Access Key": s3_secret_access_key,
        "Session ID": session_id,
        "Matrix User ID": matrix_user_id,
        "Matrix Access Token": matrix_access_token,
        "Matrix Device ID": matrix_device,
        "Matrix Keys File": matrix_keys_file,
        "Matrix Keys Passphrase": matrix_keys_passphrase,
        "Matrix Room IDs": matrix_room_ids,
        "Fetch Limit": fetch_limit,
        "Citation Limit": citation_limit,
    }

    invalid_configurations = [name for name, value in config_map.items() if not value]
    if invalid_configurations:
        st.error(f"Please fill in all required configurations: {', '.join(invalid_configurations)}")
        st.stop()

    # Prepare configurations
    app_config = AppConfig(
        cohere_api_key=cohere_api_key,
        zilliz_uri=zilliz_uri,
        zilliz_token=zilliz_token,
        zilliz_collection_name=zilliz_collection_name,
        google_api_key=google_api_key,
        db_url=db_url,
        s3_bucket=s3_bucket,
        s3_region_name=s3_region_name,
        s3_endpoint_url=s3_endpoint_url,
        s3_access_key_id=s3_access_key_id,
        s3_secret_access_key=s3_secret_access_key,
    )

    session_config = SessionConfig(
        session_id=session_id,
        matrix_user_id=matrix_user_id,
        matrix_access_token=matrix_access_token,
        matrix_device_id=matrix_device,
        matrix_keys_data=matrix_keys_file.read(),
        matrix_keys_passphrase=matrix_keys_passphrase,
        matrix_room_ids=matrix_room_ids,
        fetch_limit=fetch_limit,
        citation_limit=citation_limit,
    )

    # Wire app_container
    app_container = AppContainer()
    app_container.app_config.from_pydantic(app_config)
    app_container.session_config.from_pydantic(session_config)

    sql_store = app_container.sql_store()
    await sql_store.migrate()

    sync_use_case = await app_container.sync_use_case()
    query_use_case = app_container.query_use_case()
    document_repository = app_container.document_repository()
    vector_store = app_container.vector_store()
    file_store = app_container.file_store()

    progress_bar = st.empty()

    def progress_callback(key, text, total):
        if key not in st.session_state:
            st.session_state[key] = 0

        st.session_state[key] += 1

        with progress_bar:
            st.progress(
                value=st.session_state[key] / total,
                text=text,
            )

    # Sync and Reset
    if st.sidebar.button("Sync", use_container_width=True):
        st.session_state["fetch_message_progress"] = 0
        st.session_state["ingest_event_progress"] = 0
        await sync_use_case.execute(progress_callback)
        st.success("Sync completed.")

    if st.sidebar.button("Reset", use_container_width=True):
        await document_repository.delete_by_session(session_id)
        if vector_store.client.col:
            vector_store.client.col.drop()
        await file_store.mdelete([key async for key in file_store.yield_keys()])
        st.session_state.clear()
        st.success("State reset.")

    # Query
    query = st.text_area("Ask a query:")
    if st.button("Submit") and query:
        result = await query_use_case.execute(query)
        st.session_state["qna_result"] = result

    @st.dialog(title="Citation Details", width="large")
    def citation_details(document: Document):
        st.write(document.model_dump())

    # Display results
    if "qna_result" in st.session_state:
        res = st.session_state["qna_result"]
        st.markdown("**Response:**")
        st.write(res["response"].content)
        st.markdown("**Citations:**")
        for index, document in enumerate(res["documents"], 1):
            event_type = document.metadata["exclusion"]["event_type"]
            if st.button(label=f"[{index}] {document.metadata['retrieval_score']}"):
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


if __name__ == "__main__":
    loop.run_until_complete(main())
