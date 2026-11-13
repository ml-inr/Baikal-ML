# Angle / track-line / uncertainty model (`release_v3`)

Внешняя модель G. Plotnikova, релиз `release_v3` (коммит
`15392372e4dce61b9c8c7245fb7fffe7c8f0a30f` репозитория `plotnikovgp/Baikal-ML`,
2026-10-14). По событию предсказывает направление, точку на бесконечной линии
трека и маргинальные неопределённости угла и поперечного расстояния. Это не
энергетическая реконструкция и не фит BARS; с ними соседствует, но решает другую
задачу.

## Где лежит

```
trained_models/angle_track_uncertainty/release_v3/
  angle_track_uncertainty.pt        # веса, 12.4 MB, не в git (*.pt)
  signal_noise.ckpt                 # frozen SN-чекпоинт релиза, не в git (*.ckpt)
  angle_norm.h5                     # нормировка; в git
  classifier_norm.h5                # нормировка release-SN; в git
  MODEL_CARD.md                     # upstream model card
  README.md                         # provenance, SHA256, ограничения
  model.py                          # компактные классы модели
  sn_model.py                       # EncoderTwoHead для release-SN
  independent_2019_f_metrics.json   # метрики независимого 2019-F holdout
  reco_2020_comparison.json         # парное сравнение с ROOT reco
```

SHA256 весов: `angle_track_uncertainty.pt` — `7cb8f77c…`; `signal_noise.ckpt` —
`be9cc39b…`. Полные значения и таблица файлов — в README рядом с чекпоинтом.

## Что предсказывает

- Одна CPU-модель, 3.08M параметров; bundle `format_version=2`,
  `model_type=photon_corrected_uncertainty`.
- Backbone: 5 слоёв, 8 голов Set Transformer (`hidden=256`), поверх пяти
  нормализованных признаков `[q, t, x, y, z]` строятся 10 engineered-признаков.
- Головы: направление (3), смещение точки от центроида выбранных хитов (3), две
  лог-неопределённости, плюс frozen photon-time-residual коррекция.
- `max_seq_len=128`, `center_time=false`; нормировка bundle совпадает с
  `angle_norm.h5`. Калибровка: угол 68/95 = 2.60175/6.45668, линия
  68/95 = 1.85289/4.22721.

## Отбор хитов

Модель не классифицирует хиты. Нужны хиты, прошедшие сигнал/шум:

- **canonical mode** (по умолчанию в адаптере): канонические вероятности
  `{stem}_probs_k_nsol_labelneq0_da_hs128_k0p0001.h5`;
- **release mode**: `signal_noise.ckpt` + `classifier_norm.h5` из bundle.

`signal_noise.ckpt` — **не** наш канонический SN-чекпоинт (другой SHA256 и другая
нормировка). Canonical mode согласован с остальным пайплайном, но не воспроизводит
опубликованные числа релиза; release mode воспроизводит. Режимы пишут разные
DuckDB-файлы и не смешиваются.

Порог отбора — `prob >= threshold` (семантика релиза). Общий
`inference/shared/h5_hits.signal_mask` использует `>`; этот файл его не меняет.
Базовый кат: `p >= 0.70`, не менее 8 хитов на не менее чем 2 строках.

Опциональный gate: сначала `p >= 0.70`; для событий с `theta < 90°` модель
перезапускается на `p >= 0.50`, если низкий порог добавляет хиты и каждый
добавленный хит проходит 200 ns photon-time residual. Это тот же модель, не
ансамбль. Для gate-строк калиброванные радиусы не выдаются.

## Инференс

```
inference/angle_track_uncertainty/
  predict.py                 # потоковый адаптер: наш HDF5 -> DuckDB по event_fk
  prepare_selected_h5.py     # наш HDF5 -> prepared-HDF5 формат релиза
  predict_one_pass.py        # upstream-совместимый CSV, один проход
  predict_rethresholded.py   # upstream-совместимый CSV, gate p>=0.50
  common.py, store.py        # чтение/отбор/батчи/DuckDB
  run_predict_{mc,exp}.sh
  tests/test_parity.py
```

Команды:

```bash
bash inference/angle_track_uncertainty/run_predict_mc.sh
bash inference/angle_track_uncertainty/run_predict_exp.sh
GATE=true bash inference/angle_track_uncertainty/run_predict_mc.sh
PARTS=part_0 LIMIT=100 bash inference/angle_track_uncertainty/run_predict_mc.sh
# повторный запуск с пропуском уже готовых частей:
python -m inference.angle_track_uncertainty.predict ... --skip-done-parts
```

Выход —
`preds/release_v3@angle_track_uncertainty/{source}_{sn_mode}_p070[_gate200]_track.duckdb`
плюс `run_info.json` и `prediction_history.csv`. Схема:

```
predictions(
  event_fk, event_id, data_class, h5_part_key, h5_local_event_id,
  sn_mode, selection_op, gate_applied,
  selected_threshold, n_raw_hits, n_hits, n_strings,
  theta_deg, phi_deg, direction_x/y/z,
  anchor_x_m/y_m/z_m,
  sigma_angle_deg, angle_r68_deg, angle_r95_deg,
  sigma_anchor_m, anchor_r68_m, anchor_r95_m,
  track_point_scope, uncertainty_calibration_scope
)
```

`event_fk` — `catalog_v2.events.id`; для MC и exp используется та же логика, что
в `inference/nu_classifier/predict_{mc,exp}.py`. `n_hits`/`n_strings` — отбор на
`selected_threshold`, не на обычном 0.8. `anchor_*` — точка на бесконечной линии
в cluster-local метрах.

`track_point_scope`: `validated_single_track_MC` только для MC `nuatm`/`nue2`;
`not_validated_for_possible_multi_track_or_experiment` для `muatm` и exp.
`uncertainty_calibration_scope`: `p070_crossseason` только для неизменённых
release-SN p>=0.70 MC-строк на CPU с релизными катами; `p050_gate_unvalidated`
для gate-строк; `canonical_sn_selection_unvalidated` для canonical;
`experimental_unvalidated` для exp; `changed_threshold_or_transform_unvalidated`
иначе. Радиусы — SQL NULL там, где калибровка не заявлена.

Точное воспроизведение upstream CSV: `prepare_selected_h5.py` →
`predict_one_pass.py` / `predict_rethresholded.py`; см. README в
`inference/angle_track_uncertainty/`.

## Ограничения

- Обучение/валидация: `nuatm`, `nue2` и только **single-track** `muatm`.
  Мульти-трековые `muatm` и экспериментальные события не имеют валидированной
  точки/неопределённости.
- Точка — одна точка на бесконечной линии; концы трека и длина не
  предсказываются.
- 68%/95% радиусы калиброваны маргинально; для gate-строк и других порогов
  калибровка не заявлена.
- Экспериментальной истинной линии не существует. Экспериментальные
  неопределённости — не валидированное покрытие.
- Canonical mode наследует batch size файла probs (для `exp_full` — 512;
  известное расхождение MC/exp, `doc/sig_noise_batch_size.md`). Release SN
  использует float padding mask, поэтому и его отбор зависит от состава батча;
  batch size записывается в `run_info.json`. Точное воспроизведение релиза —
  через prepared-HDF5 путь.
- Сам angle/track backbone использует **булев** padding mask
  (`src_key_padding_mask=~full_mask`); его per-event выход от состава батча не
  зависит (различия только порядка float-редукций, ≤4e-6).

## Проверка

```bash
python inference/angle_track_uncertainty/tests/test_parity.py
```

Проверяет SHA256, загрузку модели и release-SN, конвертацию prepared-HDF5,
совпадение адаптера с one-pass CSV и gate с rethresholded CSV на синтетике.
