"""Command-line interface for the public :mod:`ms2query_db.api` workflows.

The CLI is intentionally thin: argument parsing happens here, while library
creation, index management, querying, statistics, and diagnostics are delegated
to the Python API. Every command accepts ``--dsn`` or the ``MS2QUERY_DSN``
environment variable.
"""
from __future__ import annotations

import argparse
import faulthandler
import json
import logging
import os
from dataclasses import replace
from pathlib import Path

from .api import (
    DEFAULT_INDEX_DIR,
    DEFAULT_SCHEMA,
    MS2QueryLibrary,
    create_library,
    doctor,
    library_statistics,
)
from .config import BuildConfig, QueryConfig
from .util import jsonable

logger = logging.getLogger(__name__)


def _add_connection_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--dsn",
        default=argparse.SUPPRESS,
        help="PostgreSQL connection URL. May also be set with MS2QUERY_DSN.",
    )
    parser.add_argument(
        "--schema",
        default=None,
        help=f"Library schema (default: {DEFAULT_SCHEMA}).",
    )


def _add_model_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--model",
        help="MS2DeepScore .pt or .onnx model; omit for classical-only use.",
    )
    parser.add_argument(
        "--allow-legacy",
        action="store_true",
        help=(
            "Allow legacy PyTorch model loading where supported. Only use "
            "this for a trusted model file."
        ),
    )


def parser() -> argparse.ArgumentParser:
    """Create the top-level argument parser."""
    root = argparse.ArgumentParser(
        prog="ms2query-db",
        description=(
            "Build and query auditable MS/MS reference libraries backed by "
            "PostgreSQL, pgvector, and matchms native indexes."
        ),
    )
    root.add_argument(
        "--dsn",
        default=os.environ.get("MS2QUERY_DSN"),
        help="PostgreSQL connection URL; prefer MS2QUERY_DSN for scripts.",
    )
    root.add_argument(
        "--verbose",
        action="store_true",
        help="Enable debug logging.",
    )

    commands = root.add_subparsers(
        dest="command",
        required=True,
    )

    build = commands.add_parser(
        "build",
        help="Create a new reference library.",
    )
    _add_connection_arguments(build)
    _add_model_arguments(build)
    build.add_argument("--references", required=True)
    build.add_argument("--compounds")
    build.add_argument(
        "--config",
        help="BuildConfig JSON file.",
    )
    build.add_argument(
        "--index-dir",
        default=DEFAULT_INDEX_DIR,
    )
    build.add_argument("--resume", action="store_true")
    build.add_argument(
        "--skip-native-indexes",
        action="store_true",
        help=(
            "Import PostgreSQL data only. Native classical indexes can be "
            "built later with the 'index' command."
        ),
    )

    index = commands.add_parser(
        "index",
        help="Build or rebuild local classical search indexes.",
    )
    _add_connection_arguments(index)
    index.add_argument(
        "--index-dir",
        default=DEFAULT_INDEX_DIR,
    )
    index.add_argument(
        "--methods",
        default="all",
        help="Comma-separated classical methods or 'all'.",
    )
    index.add_argument(
        "--rebuild",
        action="store_true",
        help="Archive matching local artifacts before rebuilding them.",
    )

    query = commands.add_parser(
        "query",
        help="Query one MGF file and export batch results.",
    )
    _add_connection_arguments(query)
    _add_model_arguments(query)
    query.add_argument("--input", required=True)
    query.add_argument("--output", required=True)
    query.add_argument(
        "--config",
        help="QueryConfig JSON file. Explicit CLI options override it.",
    )
    query.add_argument(
        "--index-dir",
        default=DEFAULT_INDEX_DIR,
    )
    query.add_argument("--methods")
    query.add_argument("--additional-top-n-scores")
    query.add_argument("--top-n", type=int)
    query.add_argument("--batch-size", type=int)
    query.add_argument("--classical-query-batch", type=int)
    query.add_argument("--classical-n-jobs", type=int)
    query.add_argument("--ef-search", type=int)
    query.add_argument(
        "--annotation-policy",
        choices=("validate_if_present", "require", "ignore"),
    )
    query.add_argument(
        "--ionmode-matching",
        choices=("same_mode", "cross_mode"),
    )
    query.add_argument(
        "--chemical-metric",
        choices=("minmax", "tanimoto_dot", "cosine"),
    )
    query.add_argument(
        "--aggregation-method",
        choices=("medoid", "centroid"),
    )
    query.add_argument(
        "--no-aggregation",
        action="store_true",
        help="Disable aggregation even when enabled by --config.",
    )
    query.add_argument("--aggregation-weight-source")
    query.add_argument("--aggregation-weight-power", type=float)
    query.add_argument(
        "--aggregation-selection",
        choices=("top_n", "score_delta"),
    )
    query.add_argument("--aggregation-top-n", type=int)
    query.add_argument("--aggregation-score-delta", type=float)
    query.add_argument("--chemical-search-pool", type=int)
    query.add_argument(
        "--exact-embedding-search",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    query.add_argument(
        "--exact-chemical-search",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    query.add_argument(
        "--compute-chemical-similarity",
        action=argparse.BooleanOptionalAction,
        default=None,
    )

    stats = commands.add_parser(
        "stats",
        help="Show library row and rejection counts.",
    )
    _add_connection_arguments(stats)

    diagnostics = commands.add_parser(
        "doctor",
        help="Check PostgreSQL, pgvector, and indexed matchms backends.",
    )
    _add_connection_arguments(diagnostics)

    return root


def _resolved_dsn(args, root_parser: argparse.ArgumentParser) -> str:
    dsn = getattr(args, "dsn", None) or os.environ.get("MS2QUERY_DSN")
    if not dsn:
        root_parser.error("Set MS2QUERY_DSN or provide --dsn.")
    return dsn


def _build_config(args) -> BuildConfig:
    config = (
        BuildConfig.from_file(args.config)
        if args.config
        else BuildConfig()
    )
    if args.schema:
        config = replace(config, schema=args.schema)
    if args.skip_native_indexes:
        config = replace(config, classical_methods=())
    return config


def _query_config(args) -> QueryConfig:
    config = (
        QueryConfig.from_file(args.config)
        if args.config
        else QueryConfig()
    )

    overrides = {
        "search_methods": args.methods,
        "additional_top_n_scores": args.additional_top_n_scores,
        "top_n": args.top_n,
        "batch_size": args.batch_size,
        "classical_query_batch": args.classical_query_batch,
        "classical_n_jobs": args.classical_n_jobs,
        "ef_search": args.ef_search,
        "annotation_policy": args.annotation_policy,
        "ionmode_matching": args.ionmode_matching,
        "chemical_metric": args.chemical_metric,
        "aggregation_method": args.aggregation_method,
        "aggregation_weight_source": args.aggregation_weight_source,
        "aggregation_weight_power": args.aggregation_weight_power,
        "aggregation_selection": args.aggregation_selection,
        "aggregation_top_n": args.aggregation_top_n,
        "aggregation_score_delta": args.aggregation_score_delta,
        "chemical_search_pool": args.chemical_search_pool,
        "exact_embedding_search": args.exact_embedding_search,
        "exact_chemical_search": args.exact_chemical_search,
        "compute_chemical_similarity": args.compute_chemical_similarity,
    }
    overrides = {
        key: value
        for key, value in overrides.items()
        if value is not None
    }
    if args.no_aggregation:
        overrides["aggregation_method"] = None

    if overrides:
        config = replace(config, **overrides)
    return config


def _prepare_output_directory(path: str | Path) -> Path:
    output = Path(path).expanduser().resolve()
    if output.exists():
        if not output.is_dir():
            raise FileExistsError(
                f"Query output path exists and is not a directory: {output}"
            )
        if any(output.iterdir()):
            raise FileExistsError(
                "Query output directory is not empty; choose a new directory."
            )
    else:
        output.mkdir(parents=True)
    return output


def _print_json(value) -> None:
    print(json.dumps(jsonable(value), indent=2, sort_keys=True))


def _run_build(args, dsn: str) -> None:
    config = _build_config(args)
    report = create_library(
        args.references,
        dsn,
        args.model,
        chemical_csv=args.compounds,
        config=config,
        index_dir=args.index_dir,
        resume=args.resume,
        allow_legacy=args.allow_legacy,
    )
    _print_json(report)


def _run_index(args, dsn: str) -> None:
    with MS2QueryLibrary.open(
        dsn,
        schema=args.schema or DEFAULT_SCHEMA,
        index_dir=args.index_dir,
    ) as library:
        counts = library.build_indexes(
            args.methods,
            rebuild=args.rebuild,
        )
    _print_json(counts)


def _run_query(args, dsn: str) -> None:
    config = _query_config(args)

    if (
        config.aggregation_method is not None
        and config.aggregation_weight_source == "rf_quality"
    ):
        raise ValueError(
            "The CLI cannot load an RF candidate reranker yet. Use the Python "
            "API with reranker=..., or select a spectral "
            "--aggregation-weight-source."
        )

    output = _prepare_output_directory(args.output)
    with MS2QueryLibrary.open(
        dsn,
        args.model,
        schema=args.schema or DEFAULT_SCHEMA,
        index_dir=args.index_dir,
        allow_legacy=args.allow_legacy,
    ) as library:
        for number, batch in enumerate(
            library.iter_query_file(
                args.input,
                config=config,
            )
        ):
            batch.export(output / f"batch_{number:06d}")
            logger.info(
                "Exported query batch %d: %d queries",
                number,
                len(batch.results),
            )


def main(argv=None) -> int:
    """Run the command-line interface and return a process exit code."""
    try:
        faulthandler.enable()
    except (RuntimeError, OSError):
        # Some redirected/test streams do not expose a usable file descriptor.
        pass

    root_parser = parser()
    args = root_parser.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    dsn = _resolved_dsn(args, root_parser)

    try:
        if args.command == "build":
            _run_build(args, dsn)
        elif args.command == "index":
            _run_index(args, dsn)
        elif args.command == "query":
            _run_query(args, dsn)
        elif args.command == "stats":
            _print_json(
                library_statistics(
                    dsn,
                    schema=args.schema or DEFAULT_SCHEMA,
                )
            )
        elif args.command == "doctor":
            _print_json(
                doctor(
                    dsn,
                    schema=args.schema or DEFAULT_SCHEMA,
                )
            )
        else:  # pragma: no cover - argparse guarantees a known command.
            raise AssertionError(f"Unhandled command: {args.command}")

    except KeyboardInterrupt:
        logger.error(
            "Interrupted. Resume builds with identical arguments and --resume."
        )
        return 130
    except Exception:
        logger.exception(
            "Operation stopped. No destructive reset is needed; committed "
            "build batches remain available."
        )
        return 1

    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
