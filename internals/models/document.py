from typing import Optional, Dict

from sqlalchemy import Column, JSON, TEXT
from sqlmodel import SQLModel, Field


class DocumentRecord(SQLModel, table=True):
    __tablename__ = "document_record"
    __table_args__ = {"extend_existing": True}
    id: str = Field(sa_column=Column(TEXT, primary_key=True))
    session_id: str = Field(sa_column=Column(TEXT))
    event_type: str = Field(sa_column=Column(TEXT))
    event_source: Dict = Field(sa_column=Column(JSON))
    uri: Optional[str] = Field(sa_column=Column(TEXT, nullable=True))
