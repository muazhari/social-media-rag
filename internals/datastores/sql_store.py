from typing import Dict

from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from sqlmodel import SQLModel

from internals.models.config import AppConfig


class SqlStore:
    def __init__(
            self,
            app_config: Dict,
    ):
        self.app_config = AppConfig(**app_config)
        self.db_engine = create_async_engine(url=self.app_config.db_url)
        self.async_session = async_sessionmaker(bind=self.db_engine)

    async def migrate(self):
        async with self.db_engine.begin() as connection:
            await connection.run_sync(SQLModel.metadata.create_all, checkfirst=True)
