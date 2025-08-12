from pydantic_settings import BaseSettings


class AppConfig(BaseSettings):
    db_url: str
    zilliz_uri: str
    zilliz_token: str
    zilliz_collection_name: str
    cohere_api_key: str
    google_api_key: str
    s3_bucket: str
    s3_endpoint_url: str
    s3_access_key_id: str
    s3_secret_access_key: str
    s3_region_name: str


class SessionConfig(BaseSettings):
    session_id: str
    matrix_user_id: str
    matrix_access_token: str
    matrix_device_id: str
    matrix_keys_data: bytes
    matrix_keys_passphrase: str
    matrix_room_ids: str
    fetch_limit: int
    citation_limit: int
