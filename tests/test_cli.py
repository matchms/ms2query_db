"""CLI parsing and configuration tests with repository resources."""
from __future__ import annotations

from pathlib import Path

from ms2query_db.cli import _query_config, main, parser


RESOURCES = Path(__file__).parent / "resources"
TEST_SPECTRA = RESOURCES / "test_spectra.mgf"


def _require(path: Path) -> Path:
    assert path.is_file(), f"Missing Git-tracked test resource: {path}"
    return path


def test_parser_uses_public_repository_name_and_resource_query_arguments(tmp_path):
    command = parser()
    assert command.prog == "ms2query-db"

    args = command.parse_args(
        [
            "query",
            "--input",
            str(_require(TEST_SPECTRA)),
            "--output",
            str(tmp_path / "results"),
            "--methods",
            "ms2deepscore,entropy_hybrid",
            "--top-n",
            "17",
            "--aggregation-method",
            "medoid",
            "--aggregation-weight-source",
            "entropy_hybrid",
            "--aggregation-weight-power",
            "2",
            "--aggregation-selection",
            "top_n",
            "--aggregation-top-n",
            "17",
        ]
    )

    config = _query_config(args)
    assert config.search_methods == (
        "ms2deepscore",
        "entropy_hybrid",
    )
    assert config.top_n == 17
    assert config.aggregation_method == "medoid"
    assert config.aggregation_weight_source == "entropy_hybrid"
    assert config.aggregation_weight_power == 2.0
    assert config.aggregation_selection == "top_n"
    assert config.aggregation_top_n == 17


def test_no_aggregation_overrides_query_config_file(tmp_path):
    from ms2query_db.config import QueryConfig

    config_path = tmp_path / "query.json"
    config_path.write_text(
        __import__("json").dumps(
            QueryConfig(
                search_methods=("entropy_hybrid",),
                aggregation_method="medoid",
                aggregation_weight_source="entropy_hybrid",
            ).to_dict()
        ),
        encoding="utf-8",
    )

    args = parser().parse_args(
        [
            "query",
            "--input",
            str(_require(TEST_SPECTRA)),
            "--output",
            str(tmp_path / "output"),
            "--config",
            str(config_path),
            "--no-aggregation",
        ]
    )
    config = _query_config(args)

    assert config.search_methods == ("entropy_hybrid",)
    assert config.aggregation_method is None


def test_cli_reports_that_rf_aggregation_needs_python_reranker(tmp_path):
    """Until reranker persistence exists, the CLI must fail before DB access."""
    code = main(
        [
            "--dsn",
            "postgresql://unused:unused@127.0.0.1:1/unused",
            "query",
            "--input",
            str(_require(TEST_SPECTRA)),
            "--output",
            str(tmp_path / "output"),
            "--aggregation-method",
            "medoid",
        ]
    )

    assert code == 1
