"""Parse newline-delimited key/value configuration."""


def parse_config(raw: str) -> dict[str, str]:
    """Return stripped key/value pairs."""

    result: dict[str, str] = {}
    for line in raw.splitlines():
        key, value = line.split("=", maxsplit=1)
        result[key.strip()] = value.strip()
    return result
