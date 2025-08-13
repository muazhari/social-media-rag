from dependency_injector import containers, providers
from langchain_google_genai import ChatGoogleGenerativeAI

from internals.containers.matrix_client import make_matrix_client
from internals.datastores.file_store import FileStore
from internals.datastores.sql_store import SqlStore
from internals.datastores.vector_store import VectorStore
from internals.repositories.document_repository import DocumentRepository
from internals.use_cases.query_use_case import QueryUseCase
from internals.use_cases.sync_use_case import SyncUseCase


class AppContainer(containers.DeclarativeContainer):
    app_config = providers.Configuration()
    session_config = providers.Configuration()

    # SQL store using async SQLAlchemy
    sql_store = providers.Factory(
        SqlStore,
        app_config=app_config,
    )
    # File store using S3-compatible storage
    file_store = providers.Factory(
        FileStore,
        app_config=app_config,
        session_config=session_config
    )

    # Embedding and vector store
    vector_store = providers.Factory(
        VectorStore,
        app_config=app_config,
        session_config=session_config
    )

    # Language model
    llm = providers.Factory(
        ChatGoogleGenerativeAI,
        model="gemini-2.5-pro",
        google_api_key=app_config.google_api_key
    )

    # Matrix client factory with DI
    matrix_client = providers.Factory(
        make_matrix_client,
        matrix_user_id=session_config.matrix_user_id,
        matrix_access_token=session_config.matrix_access_token,
        matrix_device_id=session_config.matrix_device_id,
        matrix_keys_data=session_config.matrix_keys_data,
        matrix_keys_passphrase=session_config.matrix_keys_passphrase
    )

    # Repository
    document_repository = providers.Factory(
        DocumentRepository,
        sql_store=sql_store,
    )

    # Use cases
    sync_use_case = providers.Factory(
        SyncUseCase,
        document_repository=document_repository,
        file_store=file_store,
        vector_store=vector_store,
        matrix_client=matrix_client,
        session_config=session_config
    )
    query_use_case = providers.Factory(
        QueryUseCase,
        document_repository=document_repository,
        file_store=file_store,
        vector_store=vector_store,
        llm=llm,
        session_config=session_config
    )
