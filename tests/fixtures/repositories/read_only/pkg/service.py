"""Small symbol-inspection fixture."""


class ParserService:
    """Wrap parser operations."""

    def parse(self, raw: str) -> str:
        return raw.strip()

    async def parse_async(self, raw: str) -> str:
        return self.parse(raw)


async def load_parser(name: str) -> ParserService:
    """Build a parser service without external I/O."""

    _ = name
    return ParserService()
