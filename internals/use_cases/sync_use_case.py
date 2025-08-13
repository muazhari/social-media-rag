import asyncio
import base64
import copy
import pickle
import uuid
from typing import List, Dict, Callable

from langchain_core.documents import Document
from nio import RoomMessageText, RoomMessageMedia, RoomEncryptedMedia, DownloadError, Event, AsyncClient, \
    RoomMessagesError
from nio.crypto import decrypt_attachment

from internals.datastores.file_store import FileStore
from internals.datastores.vector_store import VectorStore
from internals.models.config import SessionConfig
from internals.models.document import DocumentRecord
from internals.repositories.document_repository import DocumentRepository


class SyncUseCase:
    def __init__(
            self,
            document_repository: DocumentRepository,
            file_store: FileStore,
            vector_store: VectorStore,
            matrix_client: AsyncClient,
            session_config: Dict
    ):
        self.document_repository = document_repository
        self.file_store = file_store
        self.vector_store = vector_store
        self.matrix_client = matrix_client
        self.session_config = SessionConfig(**session_config)

    async def process_event(self, event: Event) -> List[Document]:
        if isinstance(event, RoomMessageText):
            doc = Document(
                id=str(uuid.uuid5(uuid.NAMESPACE_OID, event.event_id)),
                page_content=event.body,
                metadata={
                    "exclusion": {"event_type": "text", "event_source": event.source}
                }
            )
            return [doc]
        elif isinstance(event, RoomMessageMedia):
            mime = event.source["content"]["info"]["mimetype"]
            uri = await self.matrix_client.mxc_to_http(event.url)
            download_response = await self.matrix_client.download(event.url)
            if isinstance(download_response, DownloadError):
                raise Exception(f"Download failed: {download_response}")
            if mime.startswith("image/"):
                b64 = f"data:{mime};base64,{base64.b64encode(download_response.body).decode()}"
                doc = Document(
                    id=str(uuid.uuid5(uuid.NAMESPACE_OID, event.event_id)),
                    page_content=b64,
                    metadata={
                        "exclusion": {
                            "event_type": "media",
                            "uri": uri,
                            "data": download_response.body,
                            "event_source": event.source
                        }
                    }
                )
                return [doc]
            return []
        elif isinstance(event, RoomEncryptedMedia):
            mime = event.source["content"]["info"]["mimetype"]
            uri = await self.matrix_client.mxc_to_http(event.url)
            download_response = await self.matrix_client.download(event.url)
            if isinstance(download_response, DownloadError):
                raise Exception(f"Download failed: {download_response}")
            data = decrypt_attachment(
                ciphertext=download_response.body,
                key=event.key["k"],
                hash=event.hashes["sha256"],
                iv=event.iv
            )
            if mime.startswith("image/"):
                b64 = f"data:{mime};base64,{base64.b64encode(data).decode()}"
                doc = Document(
                    id=str(uuid.uuid5(uuid.NAMESPACE_OID, event.event_id)),
                    page_content=b64,
                    metadata={
                        "exclusion": {
                            "event_type": "encrypted_media",
                            "uri": uri,
                            "data": data,
                            "event_source": event.source
                        }
                    }
                )
                return [doc]
            return []
        else:
            return []

    async def ingest_event(
            self,
            event: Event,
            progress_callback: Callable[[], None],
    ):
        documents = await self.process_event(event)

        document_ids = []
        bytes_documents = []
        vector_documents = []

        for document in documents:
            # Persist file blob and vector in parallel
            bytes_document = pickle.dumps(document)
            vector_document = copy.deepcopy(document)
            del vector_document.metadata["exclusion"]
            bytes_documents.append(bytes_document)
            vector_documents.append(vector_document)
            document_ids.append(document.id)

            document_record = DocumentRecord(
                id=document.id,
                session_id=self.session_config.session_id,
                event_type=document.metadata["exclusion"]["event_type"],
                event_source=document.metadata["exclusion"]["event_source"],
                uri=document.metadata["exclusion"].get("uri")
            )
            await self.document_repository.add(document_record)

        if len(documents) > 0:
            await self.file_store.amset(list(zip(document_ids, bytes_documents)))
            if self.vector_store.client.col:
                await self.vector_store.adelete(ids=document_ids)
            await self.vector_store.aadd_documents(ids=document_ids, documents=vector_documents)

        progress_callback()

    async def fetch_message(
            self,
            room_id: str,
            limit: int,
            progress_callback: Callable[[], None],
    ):
        room_messages_response = await self.matrix_client.room_messages(
            room_id=room_id,
            limit=limit
        )
        if isinstance(room_messages_response, RoomMessagesError):
            raise Exception(f"Fetch room messages failed: {room_messages_response}")

        progress_callback()

        return room_messages_response

    async def execute(self, progress_callback: Callable[[str, str, float], None]):
        try:
            # Fetch messages concurrently
            room_ids = self.session_config.matrix_room_ids.split(",")
            fetch_tasks = []
            for room_id in room_ids:
                fetch_task = self.fetch_message(
                    room_id=room_id.strip(),
                    limit=self.session_config.fetch_limit,
                    progress_callback=lambda: progress_callback(
                        "fetch_message_progress",
                        f"Fetching messages from room {room_id.strip()}",
                        len(room_ids)
                    )
                )
                fetch_tasks.append(fetch_task)

            fetched_messages = await asyncio.gather(*fetch_tasks)

            events = []
            for fetched_message in fetched_messages:
                for event in fetched_message.chunk:
                    if not isinstance(event, (RoomMessageText, RoomMessageMedia, RoomEncryptedMedia)):
                        continue
                    events.append(event)

            ingest_tasks = []
            for event in events:
                ingest_task = self.ingest_event(
                    event=event,
                    progress_callback=lambda: progress_callback(
                        "ingest_event_progress",
                        f"Ingesting event {event.event_id}",
                        len(events)
                    )
                )
                ingest_tasks.append(ingest_task)
            await asyncio.gather(*ingest_tasks)
        finally:
            await self.matrix_client.close()
