import asyncio
from typing import List, AsyncIterator, Tuple, Dict

import aioboto3

from internals.models.config import AppConfig, SessionConfig


class FileStore:
    def __init__(
            self,
            app_config: Dict,
            session_config: Dict
    ):
        self.app_config = AppConfig(**app_config)
        self.session_config = SessionConfig(**session_config)
        self.session = aioboto3.Session(
            aws_access_key_id=self.app_config.s3_access_key_id,
            aws_secret_access_key=self.app_config.s3_secret_access_key,
            region_name=self.app_config.s3_region_name
        )

    async def amset(self, items: List[Tuple[str, bytes]]):
        """Upload multiple items in parallel."""
        async with self.session.client('s3', endpoint_url=self.app_config.s3_endpoint_url) as client:
            # ensure bucket exists
            try:
                await client.head_bucket(Bucket=self.app_config.s3_bucket)
            except client.exceptions.ClientError as e:
                code = e.response.get("Error", {}).get("Code")
                if code in ("404", "NoSuchBucket"):
                    await client.create_bucket(Bucket=self.app_config.s3_bucket)
                else:
                    raise e
            # upload objects
            tasks = []
            for key, data in items:
                path = f"{self.session_config.session_id}/{key}"
                tasks.append(client.put_object(Bucket=self.app_config.s3_bucket, Key=path, Body=data))
            await asyncio.gather(*tasks)

    async def amget(self, keys: List[str]) -> List[bytes]:
        """Download multiple items in parallel."""
        async with self.session.client('s3', endpoint_url=self.app_config.s3_endpoint_url) as client:
            tasks = []
            for key in keys:
                path = f"{self.session_config.session_id}/{key}"
                tasks.append(self._download(client, path))
            return await asyncio.gather(*tasks)

    async def _download(self, client, path: str) -> bytes:
        """Helper to download a single object."""
        response = await client.get_object(Bucket=self.app_config.s3_bucket, Key=path)
        body = response['Body']
        return await body.read()

    async def yield_keys(self) -> AsyncIterator[str]:
        """Yield all keys under the session prefix."""
        prefix = f"{self.session_config.session_id}/"
        async with self.session.client('s3', endpoint_url=self.app_config.s3_endpoint_url) as client:
            paginator = client.get_paginator('list_objects_v2')
            async for page in paginator.paginate(Bucket=self.app_config.s3_bucket, Prefix=prefix):
                for obj in page.get('Contents', []):
                    # strip prefix
                    yield obj['Key'][len(prefix):]

    async def mdelete(self, keys: List[str]):
        """Delete multiple objects individually to avoid batch delete errors."""
        async with self.session.client('s3', endpoint_url=self.app_config.s3_endpoint_url) as client:
            tasks = []
            for key in keys:
                path = f"{self.session_config.session_id}/{key}"
                tasks.append(client.delete_object(Bucket=self.app_config.s3_bucket, Key=path))
            await asyncio.gather(*tasks)
