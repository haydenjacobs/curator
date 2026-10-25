"""
Get recommendations. Run, for example:

    python recommend.py "Dune" "Neuromancer"
    python recommend.py id:12345

Each argument is a book you liked: a title (case doesn't matter) or id:<number>.
This file only handles the command line; the algorithm is in curator/engine.py.
"""

import argparse
import sys

import config
from curator.data import load_catalog
from curator.engine import EngineSettings, Recommender, RecommendationResult
from curator.errors import CuratorError


def format_result(result: RecommendationResult, decimals: int, top_n: int) -> str:
    lines = ["Because you liked:"]
    for b in result.inputs:
        lines.append(f"  • {b['title']} — {b['author'] or 'unknown author'} ({b['year'] or '?'})  [id:{b['id']}]")
    if not result.recommendations:
        lines.append("\nNo recommendations found (every candidate was excluded).")
        return "\n".join(lines)
    lines.append("")
    if len(result.recommendations) > top_n:
        lines.append(f"(Showing {len(result.recommendations)} instead of {top_n} so every book you entered is represented.)\n")
    for r in result.recommendations:
        b = r.book
        lines.append(f"{r.rank}. {b['title']} — {b['author'] or 'unknown author'} ({b['year'] or '?'})  [id:{b['id']}]")
        if len(result.inputs) > 1:
            lines.append(f"   because you liked: {'; '.join(b['title'] for b in r.because_of)}")
        lines.append(f"   similarity: {r.similarity:.{decimals}f}")
        if r.shared_tags:
            lines.append(f"   why: {', '.join(t.name for t in r.shared_tags)}")
        else:
            lines.append("   why: no single tag strongly shared (overall tag profile is similar)")
    return "\n".join(lines)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Recommend books based on books you liked.")
    parser.add_argument("books", nargs="+", help='Titles of books you liked, or id:<number>')
    parser.add_argument("-n", "--top", type=int, default=None, help=f"How many results (default {config.TOP_N})")
    args = parser.parse_args(argv)

    try:
        settings = EngineSettings.from_config(config)
        if args.top is not None:
            settings = EngineSettings(**{**settings.__dict__, "top_n": args.top})
        engine = Recommender(load_catalog(config.PROCESSED_DIR), settings)
        result = engine.recommend(args.books)
    except CuratorError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    print(format_result(result, config.DISPLAY_DECIMALS, settings.top_n))
    return 0


if __name__ == "__main__":
    sys.exit(main())
