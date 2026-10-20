"""Mode-specific, sharded native indexes for classical matchms search.

Native indexes are immutable artifacts derived from a ready PostgreSQL library.
Each ion mode and scoring method receives its own directory and manifest. A shard
bounds both index-construction memory and the dense query-by-reference score
matrix produced by matchms. Searching every shard and merging each shard's
exact top-N yields the exact global top-N for that native score; this layer does
not introduce an ANN approximation.
"""

from __future__ import annotations

import gc
import hashlib
import json
import logging
import os
import shutil
import tempfile
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from ..config import CLASSICAL_METHODS, methods_tuple
from ..errors import CompatibilityError, ContractError
from ..processing import collection_from_records
from ..util import atomic_json, digest_json, file_sha256
from .native import (
    build_index,
    index_spectrum_count,
    load_index,
    make_scorer,
    save_index,
    scorer_identity,
    search_arrays,
    top_indices,
)

logger = logging.getLogger(__name__)

NATIVE_INDEX_FORMAT_VERSION = 1
_VALID_IONMODES = ("positive", "negative")


class NativeIndexStore:
    """Build, verify, and search sharded classical spectral indexes.

    Parameters
    ----------
    database
        Open :class:`~ms2query_db.db.database.Database` for a ready library.
    root
        Root directory containing immutable native-index artifacts.
    scorer_factory
        Dependency-injection hook used primarily by tests.

    Notes
    -----
    A ``BuildConfig(ionmode='both')`` library always gets separate positive and
    negative native indexes. This is required for correct same-mode top-N
    semantics; searching one mixed index and filtering afterwards could discard
    valid same-mode hits.
    """

    def __init__(
        self,
        database,
        root,
        *,
        scorer_factory=make_scorer,
    ):
        self.database = database
        self.root = Path(root).expanduser().resolve()
        self.scorer_factory = scorer_factory
        self._descriptors: dict[tuple[str, str], dict] = {}
        self._verified: dict[str, tuple] = {}

        info = database.info()
        if info["state"] != "ready":
            raise CompatibilityError(
                "Finish database ingestion before building or searching native indexes."
            )

        self.library_id = info["library_id"]
        self.library_descriptor = info["descriptor"]
        self.config = database.config

    @property
    def ionmodes(self) -> tuple[str, ...]:
        """Actual spectrum modes represented by this library configuration."""
        if self.config.ionmode == "both":
            return _VALID_IONMODES
        return (self.config.ionmode,)

    def _resolve_ionmode(
        self,
        ionmode: str | None,
        *,
        require_explicit_for_both: bool,
    ) -> str:
        if ionmode is None:
            if self.config.ionmode == "both" and require_explicit_for_both:
                raise ValueError(
                    "A combined library requires ionmode='positive' or "
                    "ionmode='negative' for native index search."
                )
            if self.config.ionmode != "both":
                return self.config.ionmode
            raise ValueError("An ion mode must be selected for this operation.")

        mode = str(ionmode).strip().lower()
        if mode not in _VALID_IONMODES:
            raise ValueError("ionmode must be 'positive' or 'negative'.")
        if self.config.ionmode != "both" and mode != self.config.ionmode:
            raise ValueError(
                f"Library contains {self.config.ionmode!r} spectra, not {mode!r}."
            )
        return mode

    @staticmethod
    def _validate_method(method: str) -> str:
        if method not in CLASSICAL_METHODS:
            raise ValueError(
                f"Native indexes support only {CLASSICAL_METHODS}; got {method!r}."
            )
        return method

    def _identity(self, method: str, ionmode: str) -> tuple[dict, Path]:
        """Return the immutable index identity and artifact directory."""
        method = self._validate_method(method)
        mode = self._resolve_ionmode(
            ionmode,
            require_explicit_for_both=True,
        )
        key = (method, mode)

        if key not in self._descriptors:
            scorer = self.scorer_factory(
                method,
                self.config.scoring,
            )
            identity = {
                "format_version": NATIVE_INDEX_FORMAT_VERSION,
                "library_id": self.library_id,
                "library_descriptor_sha256": digest_json(self.library_descriptor),
                "ionmode": mode,
                "method": method,
                "scorer": scorer_identity(scorer),
                "settings": self.config.to_dict()["scoring"],
                "shard_spectra": self.config.index_shard_spectra,
                "shard_peaks": self.config.index_shard_peaks,
                "mz_precision": self.config.mz_precision,
            }
            self._descriptors[key] = identity

        identity = self._descriptors[key]
        directory = (
            self.root
            / self.library_id
            / mode
            / method
            / digest_json(identity)[:24]
        )
        return identity, directory

    def _iter_shards(self, ionmode: str):
        """Yield bounded reference-row shards for one actual ion mode."""
        shard: list[dict] = []
        peaks = 0
        batch_size = min(
            1000,
            self.config.index_shard_spectra,
        )

        for rows in self.database.iter_spectra(
            batch_size=batch_size,
            ionmode=ionmode,
        ):
            for row in rows:
                row_mode = str(row.get("ionmode", "")).lower()
                if row_mode != ionmode:
                    raise ContractError(
                        "Mode-filtered database iteration returned a "
                        f"{row_mode!r} spectrum while building {ionmode!r} indexes."
                    )

                would_exceed_count = (
                    len(shard) >= self.config.index_shard_spectra
                )
                would_exceed_peaks = (
                    shard
                    and peaks + int(row["n_peaks"])
                    > self.config.index_shard_peaks
                )

                if shard and (would_exceed_count or would_exceed_peaks):
                    yield shard
                    shard = []
                    peaks = 0

                shard.append(row)
                peaks += int(row["n_peaks"])

        if shard:
            yield shard

    @staticmethod
    def _artifact_files(directory: Path) -> dict[str, dict]:
        """Return size/checksum metadata for all shard payload files."""
        return {
            str(path.relative_to(directory)): {
                "sha256": file_sha256(path),
                "bytes": path.stat().st_size,
            }
            for path in sorted(directory.rglob("*"))
            if path.is_file() and path.name != "shard.json"
        }

    @staticmethod
    def _verification_stamp(
        shard_dir: Path,
        files: dict[str, dict],
    ) -> tuple:
        """Cheap stamp used to avoid rehashing unchanged index files."""
        stamp = []
        for name in sorted(files):
            path = shard_dir / name
            try:
                stat = path.stat()
            except FileNotFoundError:
                return ()
            stamp.append((name, stat.st_size, stat.st_mtime_ns))
        return tuple(stamp)

    def _verify(self, shard_dir: Path, entry: dict) -> None:
        """Verify a published shard against its manifest checksums."""
        shard_dir = shard_dir.resolve()
        files = entry.get("files")
        if not isinstance(files, dict) or not files:
            raise CompatibilityError(
                f"Native shard manifest has no payload files: {shard_dir}"
            )

        key = str(shard_dir)
        stamp = self._verification_stamp(shard_dir, files)
        if stamp and self._verified.get(key) == stamp:
            return

        for name, expected in files.items():
            path = (shard_dir / name).resolve()
            if not path.is_relative_to(shard_dir) or not path.is_file():
                raise CompatibilityError(
                    f"Invalid or missing native index file: {name}"
                )
            if path.stat().st_size != expected["bytes"]:
                raise CompatibilityError(
                    f"Native index size mismatch: {path}. Rebuild this "
                    "native index, not the PostgreSQL library."
                )
            if file_sha256(path) != expected["sha256"]:
                raise CompatibilityError(
                    f"Native index checksum mismatch: {path}. Rebuild this "
                    "native index, not the PostgreSQL library."
                )

        self._verified[key] = self._verification_stamp(
            shard_dir,
            files,
        )

    @staticmethod
    def _shard_signature(rows: list[dict], identity: dict) -> dict:
        """Bind a shard to exact database rows and peak payloads."""
        ids = np.asarray(
            [row["id"] for row in rows],
            dtype="<i8",
        )
        digest = hashlib.sha256()
        for row in rows:
            digest.update(np.asarray([int(row["id"])], dtype="<i8").tobytes())
            digest.update(np.asarray([int(row["n_peaks"])], dtype="<i8").tobytes())
            digest.update(np.asarray([float(row["precursor_mz"])], dtype="<f8").tobytes())
            digest.update(str(row["ionmode"]).encode("utf-8"))
            digest.update(bytes(row["mz_values"]))
            digest.update(bytes(row["intensity_values"]))

        return {
            "identity_sha256": digest_json(identity),
            "ids_sha256": array_sha256(ids),
            "content_sha256": digest.hexdigest(),
            "n_spectra": len(rows),
            "n_peaks": sum(int(row["n_peaks"]) for row in rows),
        }

    @staticmethod
    def _fsync_payloads(directory: Path) -> None:
        """Flush shard payload files before publishing the shard directory."""
        for path in directory.rglob("*"):
            if not path.is_file():
                continue
            with path.open("rb") as handle:
                os.fsync(handle.fileno())

    def _load_manifest(
        self,
        manifest_path: Path,
        identity: dict,
    ) -> dict:
        try:
            manifest = json.loads(
                manifest_path.read_text(encoding="utf-8")
            )
        except (OSError, json.JSONDecodeError) as exc:
            raise CompatibilityError(
                f"Could not read native index manifest: {manifest_path}"
            ) from exc

        if manifest.get("identity") != identity or manifest.get("complete") is not True:
            raise CompatibilityError(
                "Native index manifest does not match the current library, "
                "ion mode, scorer implementation, or settings."
            )
        if not isinstance(manifest.get("shards"), list):
            raise CompatibilityError("Native index manifest has invalid shards metadata.")

        directory = manifest_path.parent
        for entry in manifest["shards"]:
            if not isinstance(entry, dict) or "name" not in entry or "signature" not in entry:
                raise CompatibilityError("Native index manifest contains an invalid shard entry.")
            marker = directory / entry["name"] / "shard.json"
            if not marker.is_file():
                raise CompatibilityError(
                    f"Native shard marker is missing: {marker}"
                )
            try:
                marker_entry = json.loads(marker.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise CompatibilityError(
                    f"Could not read native shard marker: {marker}"
                ) from exc
            if marker_entry != entry:
                raise CompatibilityError(
                    f"Native shard marker disagrees with manifest: {marker}"
                )
            self._verify(marker.parent, entry)

        expected_total = sum(
            int(entry["signature"]["n_spectra"])
            for entry in manifest["shards"]
        )
        if int(manifest.get("n_spectra", -1)) != expected_total:
            raise CompatibilityError(
                "Native index manifest spectrum count does not match its shards."
            )

        return manifest

    def ensure(
        self,
        method: str,
        *,
        ionmode: str | None = None,
    ) -> tuple[dict, Path]:
        """Return a complete mode-specific index, building it if necessary."""
        mode = self._resolve_ionmode(
            ionmode,
            require_explicit_for_both=True,
        )
        identity, directory = self._identity(method, mode)
        manifest_path = directory / "manifest.json"

        if manifest_path.is_file():
            return self._load_manifest(manifest_path, identity), directory

        with self.database.build_lock():
            if manifest_path.is_file():
                return self._load_manifest(manifest_path, identity), directory

            directory.mkdir(parents=True, exist_ok=True)
            scorer = self.scorer_factory(
                method,
                self.config.scoring,
            )
            entries: list[dict] = []

            for number, rows in enumerate(self._iter_shards(mode)):
                ids = np.asarray(
                    [row["id"] for row in rows],
                    dtype=np.int64,
                )
                shard_name = f"shard_{number:06d}"
                target = directory / shard_name
                signature = self._shard_signature(rows, identity)
                marker = target / "shard.json"

                if marker.is_file():
                    try:
                        entry = json.loads(marker.read_text(encoding="utf-8"))
                    except (OSError, json.JSONDecodeError) as exc:
                        raise CompatibilityError(
                            f"Could not read native shard marker: {marker}"
                        ) from exc
                    if entry.get("signature") != signature:
                        raise CompatibilityError(
                            "Reference rows changed during native-index resumption."
                        )
                    self._verify(target, entry)
                else:
                    if target.exists():
                        raise CompatibilityError(
                            f"Incomplete unrecognized index directory {target}. "
                            "Remove only this native shard before retrying."
                        )

                    staging = Path(
                        tempfile.mkdtemp(
                            prefix=f".{shard_name}_",
                            dir=directory,
                        )
                    )
                    try:
                        logger.info(
                            "Building %s/%s shard %d: %d spectra / %d peaks",
                            mode,
                            method,
                            number,
                            len(rows),
                            signature["n_peaks"],
                        )

                        collection = collection_from_records(
                            rows,
                            mz_precision=self.config.mz_precision,
                        )
                        if collection is None:
                            raise ContractError(
                                "Native shard construction received no spectra."
                            )

                        index = build_index(
                            scorer,
                            collection,
                        )
                        save_index(
                            scorer,
                            index,
                            staging / "index.npz",
                        )
                        np.save(
                            staging / "spectrum_ids.npy",
                            ids,
                            allow_pickle=False,
                        )

                        # A persisted index must be logically loadable before it
                        # becomes visible to readers; checksums alone are not enough.
                        reloaded = load_index(
                            scorer,
                            staging / "index.npz",
                        )
                        try:
                            count = index_spectrum_count(reloaded)
                            if count is not None and count != len(ids):
                                raise ContractError(
                                    "Round-tripped native index spectrum count differs "
                                    f"from its ID map ({count} != {len(ids)})."
                                )
                        finally:
                            del reloaded

                        del index, collection
                        gc.collect()

                        entry = {
                            "name": shard_name,
                            "signature": signature,
                            "files": self._artifact_files(staging),
                        }
                        atomic_json(
                            staging / "shard.json",
                            entry,
                        )
                        self._fsync_payloads(staging)
                        os.replace(staging, target)
                    finally:
                        if staging.exists():
                            shutil.rmtree(staging)

                entries.append(entry)

            manifest = {
                "identity": identity,
                "complete": True,
                "shards": entries,
                "n_spectra": sum(
                    entry["signature"]["n_spectra"]
                    for entry in entries
                ),
            }
            atomic_json(manifest_path, manifest)
            return self._load_manifest(manifest_path, identity), directory

    def build(self, methods) -> dict[str, dict[str, int]]:
        """Build configured native indexes for every library ion mode.

        Returns
        -------
        dict
            Nested ``{ionmode: {method: n_spectra}}`` build summary.
        """
        selected = tuple(
            method
            for method in methods_tuple(methods, allow_empty=True)
            if method != "ms2deepscore"
        )
        for method in selected:
            self._validate_method(method)

        return {
            mode: {
                method: self.ensure(
                    method,
                    ionmode=mode,
                )[0]["n_spectra"]
                for method in selected
            }
            for mode in self.ionmodes
        }

    def archive(
        self,
        methods,
        *,
        ionmode: str | None = None,
    ) -> None:
        """Move selected native artifacts aside before an explicit rebuild.

        PostgreSQL data are never removed. For a combined library, omitting
        ``ionmode`` archives the selected methods for both actual ion modes.
        """
        selected = tuple(
            method
            for method in methods_tuple(methods, allow_empty=True)
            if method != "ms2deepscore"
        )
        for method in selected:
            self._validate_method(method)

        if ionmode is None:
            modes = self.ionmodes
        else:
            modes = (
                self._resolve_ionmode(
                    ionmode,
                    require_explicit_for_both=True,
                ),
            )

        for mode in modes:
            for method in selected:
                _, directory = self._identity(method, mode)
                with self.database.build_lock():
                    if directory.exists():
                        stamp = datetime.now(UTC).strftime(
                            "%Y%m%dT%H%M%S%fZ"
                        )
                        directory.rename(
                            directory.with_name(
                                f"{directory.name}.archived.{stamp}"
                            )
                        )

        self._verified.clear()

    def search(
        self,
        queries,
        method: str,
        top_n: int,
        *,
        ionmode: str | None = None,
        query_batch: int = 32,
        n_jobs: int = 1,
    ) -> list[list[dict]]:
        """Return exact global top-N native hits for a query collection.

        ``queries`` should be a mode-homogeneous ``SpectraCollection``. For a
        combined library ``ionmode`` is mandatory; :class:`QueryEngine` handles
        this routing automatically.
        """
        if isinstance(top_n, bool) or not isinstance(top_n, int) or top_n < 1:
            raise ValueError("top_n must be a positive integer.")
        if isinstance(query_batch, bool) or not isinstance(query_batch, int) or query_batch < 1:
            raise ValueError("query_batch must be a positive integer.")
        if isinstance(n_jobs, bool) or not isinstance(n_jobs, int) or n_jobs == 0 or n_jobs < -1:
            raise ValueError("n_jobs must be a positive integer or -1.")

        mode = self._resolve_ionmode(
            ionmode,
            require_explicit_for_both=True,
        )
        manifest, directory = self.ensure(
            method,
            ionmode=mode,
        )
        scorer = self.scorer_factory(
            method,
            self.config.scoring,
        )

        best: list[list[dict]] = [
            []
            for _ in range(len(queries))
        ]

        for entry in manifest["shards"]:
            folder = directory / entry["name"]
            self._verify(folder, entry)

            ids = np.load(
                folder / "spectrum_ids.npy",
                allow_pickle=False,
                mmap_mode="r",
            )
            expected_count = int(
                entry["signature"]["n_spectra"]
            )
            if ids.shape != (expected_count,):
                raise ContractError(
                    "Native index ID-map dimension mismatch."
                )

            index = load_index(
                scorer,
                folder / "index.npz",
            )
            count = index_spectrum_count(index)
            if count is not None and count != len(ids):
                raise ContractError(
                    "Loaded native index spectrum count differs from its ID map."
                )

            try:
                for first in range(0, len(queries), query_batch):
                    block = queries[first : first + query_batch]
                    jobs = (
                        n_jobs
                        if n_jobs == -1
                        else min(n_jobs, len(block))
                    )
                    scores, matches = search_arrays(
                        scorer,
                        block,
                        index,
                        expected_columns=len(ids),
                        n_jobs=jobs,
                    )

                    for offset, row in enumerate(scores):
                        selected = top_indices(
                            row,
                            ids,
                            top_n,
                        )
                        hits = [
                            {
                                "spectrum_id": int(ids[column]),
                                "score": float(row[column]),
                                "matches": (
                                    int(matches[offset, column])
                                    if matches is not None
                                    else None
                                ),
                            }
                            for column in selected
                        ]

                        merged = best[first + offset] + hits
                        merged.sort(
                            key=lambda hit: (
                                -hit["score"],
                                hit["spectrum_id"],
                            )
                        )
                        best[first + offset] = merged[:top_n]

                    del scores, matches
            finally:
                # Large native/NumPy index arrays are released by reference counting.
                # Avoid a full cyclic-GC sweep after every shard (using gc.collect()) since
                # profiling showed that those forced collections dominate a substantial part
                # of otherwise useful search time.
                del index, ids

        return best

    def score_candidates(
        self,
        query,
        candidates,
        method: str,
    ) -> list[tuple[float, int | None]]:
        """Score one query against a small candidate collection exactly.

        This is complementary scoring only: it does not retrieve from the full
        library and therefore does not assign retrieval ranks.
        """
        self._validate_method(method)
        if candidates is None or len(candidates) == 0:
            return []
        if query is None or len(query) != 1:
            raise ValueError(
                "score_candidates() requires a one-spectrum query collection."
            )

        scorer = self.scorer_factory(
            method,
            self.config.scoring,
        )
        index = build_index(
            scorer,
            candidates,
        )
        try:
            scores, matches = search_arrays(
                scorer,
                query,
                index,
                expected_columns=len(candidates),
                n_jobs=1,
            )
            return [
                (
                    float(scores[0, column]),
                    int(matches[0, column])
                    if matches is not None
                    else None,
                )
                for column in range(len(candidates))
            ]
        finally:
            del index


def array_sha256(values) -> str:
    """Hash integer ID arrays using a canonical little-endian representation."""
    return hashlib.sha256(
        np.asarray(values, dtype="<i8").tobytes()
    ).hexdigest()
