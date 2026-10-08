"""Normalize text through synchronous and asynchronous entry points."""


class ParserService:
    """Provide one deterministic text-normalization rule."""

    def parse(self, raw: str) -> str:
        return " ".join(raw.split()).lower()

    async def parse_async(self, raw: str) -> str:
        return raw
