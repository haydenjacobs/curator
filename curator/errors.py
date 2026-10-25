"""Error types. Every failure in Curator is raised loudly as one of these."""


class CuratorError(Exception):
    """Base class for all Curator errors."""


class DataValidationError(CuratorError):
    """The data does not match what we expect. The build (or load) stops."""


class BookNotFoundError(CuratorError):
    """An input book is not in the catalog."""


class AmbiguousBookError(CuratorError):
    """An input title matches several books; the user must choose one."""

    def __init__(self, query, candidates):
        self.query = query
        self.candidates = candidates  # list of dicts: id, title, author, year
        lines = [f'"{query}" matches {len(candidates)} books. Pick one by id:']
        for c in candidates:
            lines.append(f"  id:{c['id']}  {c['title']} — {c['author'] or 'unknown author'} ({c['year'] or 'unknown year'})")
        super().__init__("\n".join(lines))


class ConfigError(CuratorError):
    """A setting in config.py is invalid."""
