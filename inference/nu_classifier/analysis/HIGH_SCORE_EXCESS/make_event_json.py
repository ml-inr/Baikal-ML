"""JSON события в формате `candidates/` (для передачи коллеге).

Формат повторяет файлы
`archive_tracked/inference/nu_classifier_model/candidates/`:

    {
      "eventID": <catalog_v2.events.event_id>,
      "season": <int>, "cluster": <int>, "run": <int>,
      "pulses": {
        "0": {"mask": 0|1, "amplitude": float, "charge": float,
              "time": float, "channelID": int},
        ...
      }
    }

`mask=1` — импульс принят sig-noise моделью (`prob > 0.8`); `amplitude` и
`charge` — одно и то же поле `raw/data[:, 0]` (так в примере). Файл называется
`event_<part_key>_<local_idx>.json`, как фактические файлы candidates/.

События берутся из sidecar-файла `make_top_exp_events.py` (top-N), хиты — из
`exp_full.h5`, вероятности — из замороженного probs-файла. Перед записью скрипт
сверяет число SN-хитов и SN-струн с probs-файлом и prediction-БД.

Запуск:
    python inference/nu_classifier/analysis/HIGH_SCORE_EXCESS/make_event_json.py \
        --events-json .../figures/exp_top10_h8s3_muOnlyDA_e23_top2.json --n 2
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import sys
from pathlib import Path

import duckdb
import h5py
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

from excess import paths  # noqa: E402
from inference.shared.h5_hits import STRING_DIVISOR  # noqa: E402

SOURCE = "exp_full"
GROUP = "exp_full"
PREDS_ROOT = ROOT / "inference" / "nu_classifier" / "preds"
SN_THRESHOLD = 0.8
TOP_KEYS = ("eventID", "season", "cluster", "run", "pulses")
PULSE_KEYS = ("mask", "amplitude", "charge", "time", "channelID")


def _relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path)


def _catalog_rows(event_fks: list[int]) -> dict[int, dict]:
    """Метаданные событий: catalog_v2.events + h5_locations."""
    marks = ", ".join("?" for _ in event_fks)
    con = duckdb.connect(str(paths.CATALOG), read_only=True)
    rows = con.execute(
        f"""
        SELECT e.id AS event_fk, e.season, e.cluster, e.run, e.event_id,
               h.part_key, h.local_idx
        FROM events e
        LEFT JOIN h5_locations h ON h.event_fk = e.id
        WHERE e.id IN ({marks})
        """,
        list(event_fks),
    ).fetchall()
    con.close()
    return {
        int(row[0]): {
            "event_fk": int(row[0]),
            "season": int(row[1]),
            "cluster": int(row[2]),
            "run": int(row[3]),
            "event_id": int(row[4]),
            "part_key": row[5],
            "local_idx": None if row[6] is None else int(row[6]),
        }
        for row in rows
    }


def _prediction_rows(checkpoint_dir: str | None,
                     event_fks: list[int]) -> dict[int, dict]:
    """Скор и SN-счётчики из prediction-БД, если она есть."""
    if not checkpoint_dir:
        return {}
    db = PREDS_ROOT / checkpoint_dir / f"{SOURCE}_thr0p8.duckdb"
    if not db.exists():
        print(f"warning: нет prediction-БД {_relative(db)}")
        return {}
    marks = ", ".join("?" for _ in event_fks)
    con = duckdb.connect(str(db), read_only=True)
    rows = con.execute(
        f"""
        SELECT event_fk, score, n_sn_hits, n_sn_strings
        FROM predictions WHERE event_fk IN ({marks})
        """,
        list(event_fks),
    ).fetchall()
    con.close()
    return {
        int(row[0]): {
            "score": None if row[1] is None else float(row[1]),
            "n_sn_hits": None if row[2] is None else int(row[2]),
            "n_sn_strings": None if row[3] is None else int(row[3]),
        }
        for row in rows
    }


def _header_row(part_key: str, local_idx: int) -> tuple[int, ...]:
    """Строка header_prty: season, cluster, run, physical event_id, sec, nsec."""
    with h5py.File(paths.h5(SOURCE), "r") as raw:
        row = raw[f"{GROUP}/header_prty/{part_key}"]["data"][local_idx]
    return tuple(int(value) for value in row)


def _event_slice(part_key: str, local_idx: int,
                 threshold: float) -> tuple[np.ndarray, np.ndarray,
                                            np.ndarray, dict]:
    """Хиты события и SN-маска из замороженного probs-файла."""
    h5_path = paths.h5(SOURCE)
    probs_path = paths.probs(SOURCE)
    with h5py.File(h5_path, "r") as raw, h5py.File(probs_path, "r") as probs:
        starts = raw[f"{GROUP}/raw/ev_starts/{part_key}"]["data"]
        prob_starts = probs[f"{GROUP}/ev_starts/{part_key}"]["data"]
        if not np.array_equal(starts, prob_starts):
            raise ValueError(f"{part_key}: ev_starts в h5 и probs расходятся")
        start, stop = int(starts[local_idx]), int(starts[local_idx + 1])
        data = raw[f"{GROUP}/raw/data/{part_key}"]["data"][start:stop]
        channels = raw[f"{GROUP}/raw/channels/{part_key}"]["data"][start:stop]
        prob = probs[f"{GROUP}/probs/{part_key}"]["data"][start:stop]
        prob_channels = probs[f"{GROUP}/channels/{part_key}"]["data"][
            start:stop]
        if not np.array_equal(channels, prob_channels):
            raise ValueError(f"{part_key}: каналы в h5 и probs расходятся")
        if not (len(data) == len(channels) == len(prob)):
            raise ValueError(f"{part_key}: длины data/channels/probs различны")

        mask = prob > threshold
        stored = {
            "n_sig_hits": int(probs[f"{GROUP}/n_sig_hits_0.8/{part_key}"]
                              ["data"][local_idx]),
            "n_sig_strings": int(probs[f"{GROUP}/n_sig_strings_0.8/{part_key}"]
                                 ["data"][local_idx]),
        }
    return data, channels, mask, stored


def _build_payload(meta: dict, data: np.ndarray, channels: np.ndarray,
                   mask: np.ndarray) -> dict:
    pulses = {
        str(i): {
            "mask": int(mask[i]),
            "amplitude": float(data[i, 0]),
            "charge": float(data[i, 0]),
            "time": float(data[i, 1]),
            "channelID": int(channels[i]),
        }
        for i in range(len(data))
    }
    return {
        "eventID": int(meta["event_id"]),
        "season": int(meta["season"]),
        "cluster": int(meta["cluster"]),
        "run": int(meta["run"]),
        "pulses": pulses,
    }


def _validate(payload: dict, n_pulses: int, n_sig: int) -> None:
    if tuple(payload) != TOP_KEYS:
        raise ValueError(f"ключи JSON: {tuple(payload)}, ожидались {TOP_KEYS}")
    expected = [str(i) for i in range(n_pulses)]
    if list(payload["pulses"]) != expected:
        raise ValueError("ключи pulses не совпадают с порядком хитов")
    for key, pulse in payload["pulses"].items():
        if tuple(pulse) != PULSE_KEYS:
            raise ValueError(f"pulse[{key}]: ключи {tuple(pulse)}")
        if pulse["mask"] not in (0, 1):
            raise ValueError(f"pulse[{key}]: mask={pulse['mask']}")
        if pulse["amplitude"] != pulse["charge"]:
            raise ValueError(f"pulse[{key}]: amplitude != charge")
    got = sum(pulse["mask"] for pulse in payload["pulses"].values())
    if got != n_sig:
        raise ValueError(f"mask.sum()={got}, ожидалось {n_sig}")


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--events-json", required=True,
                    help="sidecar make_top_exp_events.py со списком events")
    ap.add_argument("--n", type=int, default=2, help="сколько первых событий")
    ap.add_argument("--out-dir", default=str(HERE / "event_json"))
    ap.add_argument("--sn-threshold", type=float, default=SN_THRESHOLD)
    args = ap.parse_args()

    selection = json.loads(Path(args.events_json).read_text())
    selected = selection["events"][:args.n]
    if not selected:
        raise SystemExit("в events-json нет событий")
    event_fks = [int(event["event_fk"]) for event in selected]
    catalog = _catalog_rows(event_fks)
    predictions = _prediction_rows(selection.get("checkpoint_dir"), event_fks)

    probs_attrs: dict = {}
    with h5py.File(paths.probs(SOURCE), "r") as probs:
        probs_attrs = {key: value for key, value in probs.attrs.items()}

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    records = []

    for event in selected:
        fk = int(event["event_fk"])
        if fk not in catalog:
            raise ValueError(f"event_fk={fk} нет в catalog_v2.events")
        meta = catalog[fk]
        if meta["part_key"] != event["part_key"]:
            raise ValueError(f"{fk}: part_key в каталоге и sidecar расходятся")
        if meta["local_idx"] != int(event["local_idx"]):
            raise ValueError(f"{fk}: local_idx в каталоге и sidecar расходятся")

        header = _header_row(meta["part_key"], meta["local_idx"])
        header_meta = {
            "season": header[0], "cluster": header[1], "run": header[2],
            "event_id": header[3],
        }
        for key, value in header_meta.items():
            if value != meta[key]:
                raise ValueError(
                    f"{fk}: header_prty[{key}]={value} != catalog {meta[key]}")

        data, channels, mask, stored = _event_slice(
            meta["part_key"], meta["local_idx"], args.sn_threshold)
        n_sig = int(mask.sum())
        n_strings = int(len(np.unique(channels[mask] // STRING_DIVISOR)))
        if args.sn_threshold == SN_THRESHOLD:
            if n_sig != stored["n_sig_hits"]:
                raise ValueError(
                    f"{fk}: SN-хитов {n_sig} != probs {stored['n_sig_hits']}")
            if n_strings != stored["n_sig_strings"]:
                raise ValueError(
                    f"{fk}: SN-струн {n_strings} != probs "
                    f"{stored['n_sig_strings']}")

        pred = predictions.get(fk, {})
        pred_checks = {}
        if pred:
            score = float(event["score"])
            if pred["score"] is not None:
                pred_checks["score_matches"] = (
                    abs(score - pred["score"]) < 1e-6)
                if not pred_checks["score_matches"]:
                    raise ValueError(f"{fk}: score в sidecar и БД расходятся")
            if pred["n_sn_hits"] is not None:
                pred_checks["n_sn_hits_matches"] = pred["n_sn_hits"] == n_sig
                if not pred_checks["n_sn_hits_matches"]:
                    raise ValueError(f"{fk}: SN-хиты не совпали с prediction-БД")
            if pred["n_sn_strings"] is not None:
                pred_checks["n_sn_strings_matches"] = (
                    pred["n_sn_strings"] == n_strings)
                if not pred_checks["n_sn_strings_matches"]:
                    raise ValueError(
                        f"{fk}: SN-струны не совпали с prediction-БД")

        payload = _build_payload(meta, data, channels, mask)
        _validate(payload, len(data), n_sig)

        name = f"event_{meta['part_key']}_{meta['local_idx']}.json"
        path = out_dir / name
        path.write_text(json.dumps(payload, indent=2))
        digest = hashlib.sha256(path.read_bytes()).hexdigest()

        # Проверка записанного файла: формат и маска читаются обратно.
        reread = json.loads(path.read_text())
        _validate(reread, len(data), n_sig)

        records.append({
            "file": name,
            "sha256": digest,
            "event_fk": fk,
            "eventID": int(meta["event_id"]),
            "season": int(meta["season"]),
            "cluster": int(meta["cluster"]),
            "run": int(meta["run"]),
            "part_key": meta["part_key"],
            "local_idx": int(meta["local_idx"]),
            "score": float(event["score"]),
            "prediction_score": pred.get("score"),
            "n_pulses": int(len(data)),
            "n_sn_signal": n_sig,
            "n_sn_strings": n_strings,
            "probs_stored": stored,
            "prediction_checks": pred_checks,
            "header_event_id_matches": (
                header_meta["event_id"] == meta["event_id"]),
        })
        print(f"written {path}  ({len(data)} импульсов, {n_sig} SN)")

    manifest = {
        "script": _relative(Path(__file__)),
        "written": datetime.datetime.now().isoformat(timespec="seconds"),
        "format": "archive_tracked/inference/nu_classifier_model/candidates/",
        "format_writer": ("archive_tracked/inference/nu_classifier_model/"
                          "analysis/extract_moe_candidates.py"),
        "format_example": ("archive_tracked/inference/nu_classifier_model/"
                           "candidates/strange_exp_events/"
                           "event_part_s2020_c01_r0188_12857.json"),
        "events_json": _relative(Path(args.events_json)),
        "events_json_written": selection.get("written"),
        "checkpoint_dir": selection.get("checkpoint_dir"),
        "prediction_checkpoint": selection.get("prediction_checkpoint"),
        "prediction_db": (
            f"inference/nu_classifier/preds/{selection.get('checkpoint_dir')}/"
            f"{SOURCE}_thr0p8.duckdb"
            if selection.get("checkpoint_dir") else None),
        "source": SOURCE,
        "selection": selection.get("selection"),
        "sn": {
            "threshold": args.sn_threshold,
            "probs_file": _relative(paths.probs(SOURCE)),
            "model": probs_attrs.get("sig_noise_model"),
            "batch_size": (int(probs_attrs["sig_noise_batch_size"])
                           if "sig_noise_batch_size" in probs_attrs else None),
            "batch_size_source": "HDF5 attr sig_noise_batch_size",
            "note": "batch size — часть идентичности отбора, а не скорость",
        },
        "event_id_convention": (
            "eventID = catalog_v2.events.event_id. Для exp_full это физический "
            "CC-идентификатор BJointHeader (sec*1e9+nsec), а не локальный номер "
            "события в ране; локальный номер — local_idx в имени файла."),
        "files": records,
        "confidence": {
            "pulse_arrays": (
                "verified: срез raw/data и raw/channels в h5; каналы сверены с "
                "probs-файлом, ev_starts совпадают"),
            "sn_mask": (
                "verified: prob > threshold; число хитов и струн сверено со "
                "stored n_sig_hits_0.8/n_sig_strings_0.8 и prediction-БД"),
            "event_id": (
                "consistent: семантика взята у писателя формата и фактических "
                "файлов candidates/; для exp_full catalog_v2 хранит физический "
                "ID (проверено по header_prty)"),
        },
    }
    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    print(f"written {manifest_path}")


if __name__ == "__main__":
    main()
