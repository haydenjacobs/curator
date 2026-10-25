"""Error types. Every failure in Curator is raised loudly as one of these."""


def describe_candidate(c: dict) -> str:
    """'id:12  Title — Creator (year)'; the year is skipped when the title already ends with it."""
    year = c["year"] or "unknown year"
    when = "" if c["title"].endswith(f"({year})") else f" ({year})"
    return f"id:{c['id']}  {c['title']} — {c['creator'] or 'unknown'}{when}"


class CuratorError(Exception):
    """Base class for all Curator errors."""


class DataValidationError(CuratorError):
    """The data does not match what we expect. The build (or load) stops."""


class ItemNotFoundError(CuratorError):
    """An input book or movie is not in its catalog."""


class AmbiguousItemError(CuratorError):
    """An input title matches several books (or movies); the user must choose one."""

    def __init__(self, query, candidates, noun="book"):
        self.query = query
        self.candidates = candidates  # list of dicts: id, title, creator, year, medium
        lines = [f'"{query}" matches {len(candidates)} {noun}s. Pick one by id:']
        for c in candidates:
            lines.append(f"  {describe_candidate(c)}")
        super().__init__("\n".join(lines))


class ConfigError(CuratorError):
    """A setting in config.py is invalid."""
