from typing import List

from langchain_core.documents import Document
from langchain_core.messages import BaseMessage
from pydantic import BaseModel


class QueryResult(BaseModel):
    response: BaseMessage
    documents: List[Document]
