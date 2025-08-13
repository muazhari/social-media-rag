from typing import Dict

from internals.customs.cohere.cohere_embeddings import CustomCohereEmbeddings
from internals.customs.milvus.milvus import CustomMilvus
from internals.models.config import AppConfig, SessionConfig
from internals.utils.async_util import make_async


class VectorStore:
    def __init__(
            self,
            app_config: Dict,
            session_config: Dict
    ):
        self.app_config = AppConfig(**app_config)
        self.session_config = SessionConfig(**session_config)
        self.embedder = CustomCohereEmbeddings(
            model="embed-v4.0",
            cohere_api_key=self.app_config.cohere_api_key
        )
        self.client = CustomMilvus(
            embedding_function=self.embedder,
            connection_args={"uri": self.app_config.zilliz_uri, "token": self.app_config.zilliz_token},
            collection_name=f"{self.app_config.zilliz_collection_name}__{self.session_config.session_id.replace('-', '_')}",
            enable_dynamic_field=True,
            index_params={"metric_type": "COSINE"},
            search_params={"metric_type": "COSINE"},
        )

    @make_async
    def asimilarity_search_with_score(self, *args, **kwargs):
        return self.client.similarity_search_with_score(*args, **kwargs)

    @make_async
    def adelete(self, *args, **kwargs):
        return self.client.delete(*args, **kwargs)

    @make_async
    def aadd_documents(self, *args, **kwargs):
        return self.client.add_documents(*args, **kwargs)
