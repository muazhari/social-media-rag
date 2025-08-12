import tempfile
from typing import Optional

from nio import AsyncClient, AsyncClientConfig


async def make_matrix_client(
        matrix_user_id: str,
        matrix_access_token: str,
        matrix_device_id: str,
        matrix_keys_data: bytes,
        matrix_keys_passphrase: str,
) -> Optional[AsyncClient]:
    client = AsyncClient(
        homeserver="https://matrix.beeper.com",
        store_path=tempfile.mkdtemp(),
        config=AsyncClientConfig(
            store_sync_tokens=True,
            encryption_enabled=True
        )
    )

    client.restore_login(
        user_id=matrix_user_id,
        device_id=matrix_device_id,
        access_token=matrix_access_token
    )

    with tempfile.NamedTemporaryFile() as tf:
        tf.write(matrix_keys_data)
        await client.import_keys(infile=tf.name, passphrase=matrix_keys_passphrase)

    return client
