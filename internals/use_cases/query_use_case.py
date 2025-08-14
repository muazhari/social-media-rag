import pickle
from typing import List, Dict, Tuple

from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, BaseMessage

from internals.datastores.file_store import FileStore
from internals.datastores.vector_store import VectorStore
from internals.models.config import SessionConfig
from internals.models.value_object import QueryResult
from internals.repositories.document_repository import DocumentRepository


class QueryUseCase:
    def __init__(
            self,
            document_repository: DocumentRepository,
            file_store: FileStore,
            vector_store: VectorStore,
            llm,
            session_config: Dict
    ):
        self.document_repository = document_repository
        self.file_store = file_store
        self.vector_store = vector_store
        self.llm = llm
        self.session_config = SessionConfig(**session_config)

    async def execute(self, query: str) -> QueryResult:
        # retrieve top k similar
        retrieved: List[Tuple[Document, float]] = await self.vector_store.asimilarity_search_with_score(
            query=query,
            k=self.session_config.citation_limit
        )

        # load cached document
        document_ids = []
        scores = []
        for document, score in retrieved:
            document_ids.append(document.metadata["pk"])
            scores.append(score)

        document_blobs = await self.file_store.amget(document_ids)

        documents: List[Document] = []
        for document_blob, score in zip(document_blobs, scores):
            cached: Document = pickle.loads(document_blob)
            cached.metadata['retrieval_score'] = score
            documents.append(cached)

        # build citations for prompt
        citation_blocks = []
        for index, document in enumerate(documents, start=1):
            event_type = document.metadata['exclusion']['event_type']
            if event_type in ['text']:
                citation_blocks.append(
                    {'type': 'text', 'text': f"<citation_{index}>{document.page_content}</citation_{index}>"}
                )
            elif event_type in ['media', 'encrypted_media']:
                mime_type = document.metadata["exclusion"]["event_source"]["content"]["info"]["mimetype"]
                if mime_type.startswith("image/"):
                    uri = document.page_content
                    citation_blocks.extend([
                        {'type': 'text', 'text': f"<citation_{index}>"},
                        {'type': 'image_url', 'image_url': uri},
                        {'type': 'text', 'text': f"</citation_{index}>"}
                    ])

        # assemble prompt
        message_content = [
            {
                'type': 'text',
                'text': "<instruction>Answer the query using only the provided citations. Include citation numbers, i.e., [1, 2, 3, etc.]. If the query is unanswerable, then say the reason with an explanation.</instruction>"
            },
            {'type': 'text', 'text': f"<query>{query}</query>"},
            {'type': 'text', 'text': '<citations>'},
            *citation_blocks,
            {'type': 'text', 'text': '</citations>'}
        ]
        prompt = HumanMessage(content=message_content)

        # generate
        response: BaseMessage = await self.llm.ainvoke([prompt])

        # build result
        result = QueryResult(
            response=response,
            documents=documents
        )

        return result
