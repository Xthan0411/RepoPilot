"""Behavior checks for the config parser fixture."""

import unittest

from config_parser import parse_config


class ConfigParserTests(unittest.TestCase):
    def test_regular_key_value_pairs(self) -> None:
        self.assertEqual(
            parse_config("host = localhost\nport=5432"),
            {
                "host": "localhost",
                "port": "5432",
            },
        )

    def test_blank_configuration_is_empty(self) -> None:
        self.assertEqual(parse_config(" \n\t"), {})


if __name__ == "__main__":
    unittest.main()
