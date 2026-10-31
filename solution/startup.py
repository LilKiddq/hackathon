"""Host-side setup work that does not need to happen: the dataset map's fingerprint and a second
scan of the installed packages.

Soup tokenizes the training rows with `Dataset.from_list(rows).map(checked_format_row, ...)`
(`soup_cli.trainer.sft._map_text_sft_rows`). Before mapping, `datasets` fingerprints the
transform for its cache by serialising it with dill, and the transform closes over the tokenizer
(151k-entry vocabulary and merges): that hash takes ~2 s on Kaggle's CPU, longer than the
tokenization itself (~0.5 s for 256 rows). The dataset lives in memory (from_list), so no cache
file is ever looked up with that fingerprint.

Here the same function maps the same rows with the same transform, and the fingerprint is given
instead of computed: a SHA-256 of the rows and the settings, which is what it identifies. The
tokenized dataset, its order and every value are unchanged.
"""

import hashlib
import importlib.metadata
import json


def _fingerprint(rows, split, max_length):
    digest = hashlib.sha256()
    digest.update(f"soup-sft-rows|{split}|{max_length}|".encode())
    digest.update(json.dumps(rows, sort_keys=True, ensure_ascii=False, default=str).encode())
    return digest.hexdigest()[:32]


def patch_soup_row_mapping():
    from soup_cli.trainer import sft

    if getattr(sft._map_text_sft_rows, "_kc_patched", False):
        return
    original = sft._map_text_sft_rows

    def _map_text_sft_rows(rows, *, format_row, split, max_length):
        """Soup's `_map_text_sft_rows` (soup-cli 0.77.0) with a given fingerprint."""
        from datasets import Dataset

        from soup_cli.data.loss_mask import NoCausalLossTargetError, ensure_causal_loss_target

        try:
            fingerprint = _fingerprint(rows, split, max_length)
        except Exception:  # noqa: BLE001 - rows that json cannot dump: Soup's own path
            return original(rows, format_row=format_row, split=split, max_length=max_length)

        def checked_format_row(example: dict, row_index: int) -> dict:
            try:
                formatted = format_row(example)
                labels = formatted.get("labels")
                if labels is not None:
                    ensure_causal_loss_target(labels, max_length=max_length)
                return formatted
            except NoCausalLossTargetError as exc:
                raise ValueError(f"{split} row {row_index + 1}: {exc}") from exc

        return Dataset.from_list(rows).map(
            checked_format_row,
            with_indices=True,
            remove_columns=["messages"],
            new_fingerprint=fingerprint,
        )

    _map_text_sft_rows._kc_patched = True
    sft._map_text_sft_rows = _map_text_sft_rows


def reuse_package_index():
    """trl, like transformers, builds `importlib.metadata.packages_distributions()` when it is
    imported: a scan of every installed package's metadata, ~1-2 s on Kaggle's image with ~600
    packages. transformers has already built it (the harness imports transformers before
    apply()), and nothing is installed in between, so trl gets a copy of that same table."""
    try:
        from transformers.utils import import_utils
    except Exception:  # noqa: BLE001 - no table to reuse: trl scans as usual
        return
    table = getattr(import_utils, "PACKAGE_DISTRIBUTION_MAPPING", None)
    current = importlib.metadata.packages_distributions
    if not isinstance(table, dict) or getattr(current, "_kc_patched", False):
        return

    def packages_distributions():
        return {name: list(dists) for name, dists in table.items()}

    packages_distributions._kc_patched = True
    importlib.metadata.packages_distributions = packages_distributions
