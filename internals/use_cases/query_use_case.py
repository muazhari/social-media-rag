import pickle
from typing import List, Dict, Tuple

from langchain_core.documents import Document as LCDocument
from langchain_core.messages import HumanMessage, BaseMessage

from internals.customs.milvus.milvus import CustomMilvus
from internals.datastores.file_store import FileStore
from internals.models.config import SessionConfig
from internals.repositories.document_repository import DocumentRepository


class QueryUseCase:
    def __init__(
            self,
            document_repository: DocumentRepository,
            file_store: FileStore,
            vector_store: CustomMilvus,
            llm,
            session_config: Dict
    ):
        self.document_repository = document_repository
        self.file_store = file_store
        self.vector_store = vector_store
        self.llm = llm
        self.session_config = SessionConfig(**session_config)

    async def execute(self, query: str) -> Dict[str, any]:
        # retrieve top k similar
        retrieved: List[Tuple[LCDocument, float]] = await self.vector_store.asimilarity_search_with_score(
            query=query,
            k=self.session_config.citation_limit
        )
        docs: List[LCDocument] = []
        for doc, score in retrieved:
            # load cached document
            blob = (await self.file_store.amget([doc.id]))[0]
            cached: LCDocument = pickle.loads(blob)
            cached.metadata['retrieval_score'] = score
            docs.append(cached)

        # build citations for prompt
        citation_blocks = []
        for idx, document in enumerate(docs, start=1):
            ev_meta = document.metadata['exclusion']
            ev_type = ev_meta['event_type']
            if ev_type == 'text':
                citation_blocks.append(
                    {'type': 'text', 'text': f"<citation_{idx}>\n{document.page_content}\n</citation_{idx}>"})
            elif ev_type in ('media', 'encrypted_media'):
                uri = document.metadata['exclusion'].get('uri') or document.page_content
                citation_blocks.extend([
                    {'type': 'text', 'text': f"<citation_{idx}>"},
                    {'type': 'image_url', 'image_url': uri},
                    {'type': 'text', 'text': f"</citation_{idx}>"}
                ])
            else:
                continue

        # assemble prompt
        message_content = [
            {'type': 'text',
             'text': "<instruction>Answer the query using only provided citations. Include citation numbers. If missing info, respond 'I don't have enough information to answer that query.'</instruction>"},
            {'type': 'text', 'text': f"<query>\n{query}\n</query>"},
            {'type': 'text', 'text': '<citations>'},
            *citation_blocks,
            {'type': 'text', 'text': '</citations>'}
        ]
        prompt = HumanMessage(content=message_content)

        # generate
        response: BaseMessage = await self.llm.ainvoke([prompt])
        return {'response': response, 'documents': docs}
