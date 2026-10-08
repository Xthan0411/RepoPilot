"""Behavior checks for the parser service fixture."""

import unittest

from parser_service import ParserService


class ParserServiceTests(unittest.IsolatedAsyncioTestCase):
    async def test_async_parse_matches_sync_normalization(self) -> None:
        service = ParserService()

        self.assertEqual(await service.parse_async("  Mixed   SPACING  "), "mixed spacing")

    async def test_sync_and_async_results_match(self) -> None:
        service = ParserService()
        raw = "  Another   VALUE "

        self.assertEqual(await service.parse_async(raw), service.parse(raw))


if __name__ == "__main__":
    unittest.main()
