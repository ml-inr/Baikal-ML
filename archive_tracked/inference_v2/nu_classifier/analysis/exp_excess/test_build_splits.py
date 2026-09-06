"""Checks on the training/reference split.

Two properties matter. A part may never contribute to both reference and test, or the two
sets stop being independent. And every event the model trained on must be flagged, because
that flag — not the part it came from — is what keeps biased scores out of a measurement.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import duckdb
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

from archive_tracked.inference_v2.nu_classifier.analysis.exp_excess.build_splits import build_splits, part_is_reference


def test_reference_membership_is_stable_and_unbiased() -> bool:
    """Hashing must give the requested fraction and never depend on context."""
    parts = [f"part_{index}" for index in range(20000)]
    chosen = [part_is_reference("muatm_2020", part, 0.05, seed=42) for part in parts]
    fraction = sum(chosen) / len(parts)
    fraction_is_right = abs(fraction - 0.05) < 0.005

    # the same part decided again, in isolation, must give the same answer
    repeated = all(part_is_reference("muatm_2020", part, 0.05, seed=42) == was_chosen
                   for part, was_chosen in list(zip(parts, chosen))[:500])
    # a different class must not inherit the same decision
    class_matters = any(part_is_reference("nuatm_2020", part, 0.05, seed=42) != was_chosen
                        for part, was_chosen in list(zip(parts, chosen))[:500])
    never_chosen_at_zero = not any(part_is_reference("muatm_2020", part, 0.0, seed=42)
                                   for part in parts[:500])

    print(f"  requested 5%, got {100 * fraction:.2f}%          "
          f"{'ok' if fraction_is_right else 'FAIL'}")
    print(f"  decision independent of context           {'ok' if repeated else 'FAIL'}")
    print(f"  class name changes the decision           {'ok' if class_matters else 'FAIL'}")
    print(f"  zero fraction reserves nothing            "
          f"{'ok' if never_chosen_at_zero else 'FAIL'}")
    return fraction_is_right and repeated and class_matters and never_chosen_at_zero


def _make_fake_databases(directory: Path) -> tuple[str, str, pd.DataFrame]:
    """A predictions DB and a catalog holding four parts of one class, five events each."""
    events = pd.DataFrame([
        {"event_fk": part_index * 10 + local_idx,
         "data_class": "muatm_2020",
         "part_key": f"part_{part_index}",
         "local_idx": local_idx}
        for part_index in range(4) for local_idx in range(5)
    ])

    catalog_path = str(directory / "catalog.duckdb")
    catalog = duckdb.connect(catalog_path)
    catalog.execute("CREATE TABLE events (id BIGINT, data_class VARCHAR)")
    catalog.execute("CREATE TABLE h5_locations (event_fk BIGINT, part_key VARCHAR, "
                    "local_idx BIGINT)")
    catalog.register("source", events)
    catalog.execute("INSERT INTO events SELECT event_fk, data_class FROM source")
    catalog.execute("INSERT INTO h5_locations SELECT event_fk, part_key, local_idx FROM source")
    catalog.close()

    predictions_path = str(directory / "predictions.duckdb")
    predictions = duckdb.connect(predictions_path)
    predictions.execute("CREATE TABLE predictions (event_fk BIGINT, score FLOAT)")
    predictions.register("source", events)
    predictions.execute("INSERT INTO predictions SELECT event_fk, 0.5 FROM source")
    predictions.close()
    return predictions_path, catalog_path, events


def test_training_events_are_excluded_event_by_event(directory: Path) -> bool:
    """A part that gave events to training may still serve as reference — minus those events.

    Excluding the whole part instead would discard the events the model never saw, which is
    most of them: in nue2 the training selection took only 18% of each part it touched.
    """
    predictions_path, catalog_path, events = _make_fake_databases(directory)
    training = pd.DataFrame([{"data_class": "muatm_2020", "part_key": "part_1",
                              "local_idx": 0},
                             {"data_class": "muatm_2020", "part_key": "part_1",
                              "local_idx": 1}])
    config = {"random_seed": 42,
              "reference_part_fraction": {"muatm_2020": 1.0}}

    result = build_splits(predictions_path, catalog_path, config, training, None)

    partly_trained_part = result[result.part_key == "part_1"]
    still_reference = (partly_trained_part.role == "reference").all()
    two_events_flagged = int(partly_trained_part.used_for_labels.sum()) == 2
    clean_events_kept = int((~partly_trained_part.used_for_labels).sum()) == 3

    usable_reference = result[(result.role == "reference") & ~result.used_for_labels]
    no_training_event_survives = not usable_reference.used_for_labels.any()

    one_role_per_part = all(len(set(group.role)) == 1
                            for _, group in result.groupby("part_key"))

    print(f"\n  part that gave events to training is usable  "
          f"{'ok' if still_reference else 'FAIL'}")
    print(f"  its two training events are flagged         "
          f"{'ok' if two_events_flagged else 'FAIL'}")
    print(f"  its three clean events are kept             "
          f"{'ok' if clean_events_kept else 'FAIL'}")
    print(f"  reference-minus-training holds no training  "
          f"{'ok' if no_training_event_survives else 'FAIL'}")
    print(f"  every part has a single role                "
          f"{'ok' if one_role_per_part else 'FAIL'}")

    stored = duckdb.connect(predictions_path, read_only=True)
    row_count = stored.execute("SELECT count(*) FROM splits").fetchone()[0]
    stored.close()
    table_written = row_count == len(events)
    print(f"  table written with every event              "
          f"{'ok' if table_written else 'FAIL'}")

    return all([still_reference, two_events_flagged, clean_events_kept,
                no_training_event_survives, one_role_per_part, table_written])


def test_only_verified_exclusions_are_applied() -> bool:
    """A refuted exclusion must stay in the config and remove nothing."""
    from archive_tracked.inference_v2.nu_classifier.analysis.exp_excess.build_splits import verified_excluded_runs

    config = {"exclusions": {
        "runs": [{"key": "run_a", "status": "verified"},
                 {"key": "run_b", "status": "refuted"},
                 {"key": "run_c", "status": "inherited"}],
        "clusters": [{"key": 4, "status": "refuted"}]}}
    applied = verified_excluded_runs(config)
    only_verified = applied == {"run_a"}
    empty_config_safe = verified_excluded_runs({}) == set()
    print(f"\n  only the verified run is excluded: {sorted(applied)}      "
          f"{'ok' if only_verified else 'FAIL'}")
    print(f"  a config without exclusions removes nothing        "
          f"{'ok' if empty_config_safe else 'FAIL'}")
    return only_verified and empty_config_safe


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as directory:
        outcomes = [test_reference_membership_is_stable_and_unbiased(),
                    test_only_verified_exclusions_are_applied(),
                    test_training_events_are_excluded_event_by_event(Path(directory))]
    print("\n" + ("ALL PASSED" if all(outcomes) else "FAILED"))
    sys.exit(0 if all(outcomes) else 1)
