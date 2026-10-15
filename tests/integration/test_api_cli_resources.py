"""Optional end-to-end API/CLI tests using the tiny repository resources.

Set MS2QUERY_TEST_DSN to a disposable PostgreSQL database with pgvector. Each
case creates a random schema and removes it afterwards.
"""
from __future__ import annotations

import csv
import json
import os
from pathlib import Path
import uuid

import pytest
from sqlalchemy import text

from ms2query_db.api import (
    MS2QueryLibrary,
    create_library,
    doctor,
    library_statistics,
)
from ms2query_db.cli import main
from ms2query_db.config import BuildConfig, QueryConfig
from ms2query_db.db import Database
from ms2query_db.io import discover_sources, iter_source_batches, load_mgf_batch


RESOURCES = Path(__file__).parents[1] / "resources"
TEST_SPECTRA = RESOURCES / "test_spectra.mgf"
TEST_COMPOUNDS = RESOURCES / "test_compounds.csv"
TEST_MODEL = RESOURCES / "testmodel_v1.pt"


def _require(path: Path) -> Path:
    assert path.is_file(), f"Missing Git-tracked test resource: {path}"
    return path


def _dsn() -> str:
    value = os.environ.get("MS2QUERY_TEST_DSN")
    if not value:
        pytest.skip("Set MS2QUERY_TEST_DSN for PostgreSQL/pgvector integration tests")
    return value


def _schema(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


def _drop_schema(dsn: str, schema: str) -> None:
    database = Database(dsn, schema)
    try:
        quoted = database.engine.dialect.identifier_preparer.quote_schema(schema)
        with database.engine.begin() as connection:
            connection.execute(text(f"DROP SCHEMA IF EXISTS {quoted} CASCADE"))
    finally:
        database.close()


def _resource_ionmode() -> str:
    source = discover_sources(_require(TEST_SPECTRA))[0]
    modes = set()
    for records in iter_source_batches(source, 64):
        loaded = load_mgf_batch(records)
        if loaded.collection is None:
            continue
        for row in range(len(loaded.collection)):
            mode = str(loaded.collection[row].get("ionmode") or "").strip().lower()
            if mode in {"positive", "negative"}:
                modes.add(mode)
    if modes == {"positive", "negative"}:
        return "both"
    if len(modes) == 1:
        return next(iter(modes))
    raise AssertionError(f"Resource spectra need canonical ionmode metadata; found {modes}")


def _first_smiles() -> str:
    with _require(TEST_COMPOUNDS).open(encoding="utf-8-sig", newline="") as handle:
        row = next(csv.DictReader(handle))
    return row["SMILES"]


def test_public_api_build_open_query_and_compound_search(tmp_path):
    dsn = _dsn()
    schema = _schema("api_resource")
    index_dir = tmp_path / "indexes"
    config = BuildConfig(
        schema=schema,
        ionmode=_resource_ionmode(),
        classical_methods=(),
        batch_size=16,
    )

    try:
        report = create_library(
            _require(TEST_SPECTRA),
            dsn,
            _require(TEST_MODEL),
            chemical_csv=_require(TEST_COMPOUNDS),
            config=config,
            index_dir=index_dir,
            allow_legacy=True,
        )
        assert report.schema == schema
        assert report.counts["spectra"] > 0
        assert report.counts["compounds"] >= 100

        stats = library_statistics(dsn, schema=schema)
        assert stats["library_id"] == report.library_id
        assert stats["state"] == "ready"
        assert stats["spectra"] == report.counts["spectra"]

        diagnostics = doctor(dsn, schema=schema)
        assert diagnostics["pgvector"]
        assert diagnostics["database"]
        assert "packages" in diagnostics

        with MS2QueryLibrary.open(
            dsn,
            _require(TEST_MODEL),
            schema=schema,
            index_dir=index_dir,
            allow_legacy=True,
        ) as library:
            query_result = library.query_file(
                TEST_SPECTRA,
                config=QueryConfig(
                    search_methods=("ms2deepscore",),
                    top_n=3,
                    batch_size=4,
                    annotation_policy="ignore",
                ),
            )
            ok = [result for result in query_result.results if result.status == "ok"]
            assert ok
            assert all(len(result.candidates) >= 1 for result in ok)
            assert all(
                set(result.candidates["candidate_ionmode"]) == {result.query_ionmode}
                for result in ok
            )

            neighbors = library.search_compounds(
                _first_smiles(),
                top_n=1,
                exact=True,
            )
            assert len(neighbors) == 1
            assert neighbors.iloc[0]["chemical_similarity"] == pytest.approx(1.0)

            spectrum_id = int(ok[0].candidates.iloc[0]["spectrum_id"])
            collection = library.get_spectra([spectrum_id, spectrum_id])
            assert collection is not None
            assert len(collection) == 2
    finally:
        _drop_schema(dsn, schema)


def test_cli_build_stats_and_query_with_real_resources(tmp_path, capsys):
    dsn = _dsn()
    schema = _schema("cli_resource")
    index_dir = tmp_path / "indexes"
    output = tmp_path / "query_output"
    build_config = BuildConfig(
        schema=schema,
        ionmode=_resource_ionmode(),
        classical_methods=(),
        batch_size=16,
    )
    config_path = tmp_path / "build.json"
    config_path.write_text(json.dumps(build_config.to_dict()), encoding="utf-8")

    try:
        build_code = main(
            [
                "--dsn",
                dsn,
                "build",
                "--references",
                str(_require(TEST_SPECTRA)),
                "--compounds",
                str(_require(TEST_COMPOUNDS)),
                "--model",
                str(_require(TEST_MODEL)),
                "--allow-legacy",
                "--config",
                str(config_path),
                "--index-dir",
                str(index_dir),
                "--skip-native-indexes",
            ]
        )
        assert build_code == 0
        build_payload = json.loads(capsys.readouterr().out)
        assert build_payload["schema"] == schema

        stats_code = main(
            [
                "--dsn",
                dsn,
                "stats",
                "--schema",
                schema,
            ]
        )
        assert stats_code == 0
        stats_payload = json.loads(capsys.readouterr().out)
        assert stats_payload["state"] == "ready"
        assert stats_payload["spectra"] > 0

        query_code = main(
            [
                "--dsn",
                dsn,
                "query",
                "--schema",
                schema,
                "--input",
                str(TEST_SPECTRA),
                "--output",
                str(output),
                "--model",
                str(TEST_MODEL),
                "--allow-legacy",
                "--methods",
                "ms2deepscore",
                "--annotation-policy",
                "ignore",
                "--top-n",
                "3",
                "--batch-size",
                "4",
                "--index-dir",
                str(index_dir),
            ]
        )
        assert query_code == 0
        batch_dirs = sorted(output.glob("batch_*"))
        assert batch_dirs
        assert all((directory / "queries.csv.gz").is_file() for directory in batch_dirs)
        assert all((directory / "manifest.json").is_file() for directory in batch_dirs)
    finally:
        _drop_schema(dsn, schema)
