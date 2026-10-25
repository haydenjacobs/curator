"""
Get recommendations. Run, for example:

    python recommend.py "Dune" "Neuromancer"
    python recommend.py "The Hunger Games" --movie "Heat" --want mixed
    python recommend.py id:12345 --movie id:6789 --want movies

Positional arguments are books you liked; each --movie is a movie you liked.
Either can be a title (case doesn't matter) or id:<number>. --want picks what
comes back: books, movies, or a mixed list.
This file only handles the command line; the algorithm is in curator/engine.py.
"""

import argparse
import sys

import config
from curator.data import load_library
from curator.engine import EngineSettings, Recommender, RecommendationResult
from curator.errors import CuratorError

WANT_HEADINGS = {"books": "Books for you", "movies": "Movies for you", "mixed": "Books and movies for you"}


def describe(item: dict) -> str:
    """'Title — Creator (year)  [id:..]'; the year is skipped when the title already ends with it."""
    who = item["creator"] or ("unknown author" if item["medium"] == "book" else "unknown director")
    if item["medium"] == "movie":
        who = f"dir. {who}" if item["creator"] else who
    year = item["year"] or "?"
    when = "" if item["title"].endswith(f"({year})") else f" ({year})"
    return f"{item['title']} — {who}{when}  [id:{item['id']}]"


def format_result(result: RecommendationResult, decimals: int, top_n: int) -> str:
    lines = ["Because you liked:"]
    for b in result.inputs:
        lines.append(f"  • [{b['medium']}] {describe(b)}")

    if result.mode == "common":
        lines.append("")
        lines.append("What these have in common (strongest first):" if len(result.inputs) > 1
                     else "What stands out about it (strongest first):")
        lines.append("  " + ", ".join(t.name for t in result.common_tags))
        if result.shared_space:
            lines.append("  (Books and movies are compared on the tags both datasets have.)")

    if not result.recommendations:
        lines.append("\nNo recommendations found (every candidate was excluded).")
        return "\n".join(lines)
    lines.append("")
    lines.append(f"{WANT_HEADINGS[result.want]}:")
    if len(result.recommendations) > top_n:
        lines.append(f"(Showing {len(result.recommendations)} instead of {top_n} so every book you entered is represented.)")
    for r in result.recommendations:
        mark = f"[{r.item['medium']}] " if result.want == "mixed" else ""
        lines.append(f"{r.rank}. {mark}{describe(r.item)}")
        if result.mode == "common":
            lines.append(f"   match: {r.score:.{decimals}f}")
        else:
            if len(result.inputs) > 1:
                lines.append(f"   because you liked: {'; '.join(b['title'] for b in r.because_of)}")
            lines.append(f"   similarity: {r.score:.{decimals}f}")
        if r.shared_tags:
            lines.append(f"   why: {', '.join(t.name for t in r.shared_tags)}")
        else:
            lines.append("   why: no single tag strongly shared (overall tag profile is similar)")
    return "\n".join(lines)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Recommend books and movies based on books and movies you liked.")
    parser.add_argument("books", nargs="*", help="Titles of books you liked, or id:<number>")
    parser.add_argument("-m", "--movie", action="append", default=[], metavar="MOVIE",
                        help="A movie you liked (title or id:<number>). Repeat for several movies.")
    parser.add_argument("-w", "--want", choices=["books", "movies", "mixed"], default=config.DEFAULT_WANT,
                        help=f"What to recommend (default {config.DEFAULT_WANT})")
    parser.add_argument("-n", "--top", type=int, default=None, help=f"How many results (default {config.TOP_N})")
    args = parser.parse_args(argv)
    if not args.books and not args.movie:
        parser.error("give at least one book (e.g. \"Dune\") or movie (e.g. --movie \"Heat\")")

    try:
        settings = EngineSettings.from_config(config)
        if args.top is not None:
            settings = EngineSettings(**{**settings.__dict__, "top_n": args.top})
        engine = Recommender(load_library(config.PROCESSED_DIR), settings)
        result = engine.recommend(args.books, args.movie, args.want)
    except CuratorError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    print(format_result(result, config.DISPLAY_DECIMALS, settings.top_n))
    return 0


if __name__ == "__main__":
    sys.exit(main())
