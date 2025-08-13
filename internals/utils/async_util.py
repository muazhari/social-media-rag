import asyncio
import functools
from typing import Callable, Any


def make_async(func: Callable) -> Any:
    @functools.wraps(func)
    async def async_wrapper(*args, **kwargs) -> Any:
        return await asyncio.to_thread(functools.partial(func, *args, **kwargs))

    return async_wrapper
