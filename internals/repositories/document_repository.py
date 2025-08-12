from typing import List

from sqlmodel import select, delete

from internals.datastores.sql_store import SqlStore
from internals.models.document import DocumentRecord


class DocumentRepository:
    def __init__(self, sql_store: SqlStore):
        self.sql_store = sql_store

    async def add(self, record: DocumentRecord) -> DocumentRecord:
        async with self.sql_store.async_session() as session:
            found_record = await session.get(DocumentRecord, record.id)
            if found_record:
                found_record.session_id = record.session_id
                found_record.event_type = record.event_type
                found_record.event_source = record.event_source
                found_record.uri = record.uri
                session.add(found_record)
                record = found_record
            else:
                session.add(record)

            await session.commit()
            await session.refresh(record)
            return record

    async def get_by_session(self, session_id: str) -> List[DocumentRecord]:
        async with self.sql_store.async_session() as session:
            stmt = select(DocumentRecord).where(DocumentRecord.session_id == session_id)  # type: ignore
            result = await session.execute(stmt)
            return list(result.scalars().all())

    async def get_by_ids(self, session_id: str, ids: List[str]) -> List[DocumentRecord]:
        async with self.sql_store.async_session() as session:
            stmt = select(DocumentRecord).where(
                DocumentRecord.session_id == session_id,  # type: ignore
                DocumentRecord.id.in_(ids)  # type: ignore
            )
            result = await session.execute(stmt)
            return list(result.scalars().all())

    async def delete_by_session(self, session_id: str) -> None:
        async with self.sql_store.async_session() as session:
            stmt = delete(DocumentRecord).where(DocumentRecord.session_id == session_id)  # type: ignore
            await session.execute(stmt)
            await session.commit()
