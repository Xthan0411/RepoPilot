"""Module docs."""

import os

@decorator
class Service:
    class Nested:
        pass

    @classmethod
    def build(cls) -> "Service":
        return cls()

    async def run(self) -> str:
        async def inner() -> str:
            return "ok"

        return await inner()


def helper(value: int) -> int:
    return value + 1

CONSTANT = os.name
