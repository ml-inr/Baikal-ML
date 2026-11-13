# Расположение SN-модели (перенос 2026-11-08)

Внешняя модель сигнал/шум (SN, автор G. Plotnikov) перенесена из корня проекта
в `trained_models/sig_noise/`, а скрипты генерации h5-вероятностей — в
`inference/sig_noise/`. Это запись о переносе и о том, как он совместим со старым
кодом; сами данные, веса и формат выходных файлов не менялись.

## Куда переехало

| Было | Стало |
|---|---|
| `gplotnikov_sig_noise_models/` | `trained_models/sig_noise/` (все варианты, `archive/`, `logs/`, `test.ipynb`) |
| `gplotnikov_sig_noise_models/k_nsol_labelneq0_da_hs128_k0p0001/predict_mc_h5.py` | `inference/sig_noise/predict_mc_h5.py` |
| то же `predict_exp_h5.py` | `inference/sig_noise/predict_exp_h5.py` |

Каноническая модель — `k_nsol_labelneq0_da_hs128_k0p0001`. В её папке лежат
чекпоинт, конфиг и код модели:

```
trained_models/sig_noise/k_nsol_labelneq0_da_hs128_k0p0001/
  best_mc_2020.ckpt          # не в git (*.ckpt), кладётся вручную
  train_config_mc_2020.yaml
  sig_noise_model_v3.py      # load_model() + predict_flat()
  model_simplified.py
  encoder.py
  layers.py
```

Неканонические варианты (`nn_aug_highq_*` и др.) перенесены целиком и сохранили
свой статус в `.gitignore`: `nn_aug_highq_*` отслеживаются, остальные варианты,
`archive/` и `logs/` — игнорируются.

## Обратная совместимость

Старый import path продолжает работать. Единственный файл, оставшийся в старом
корне, — `gplotnikov_sig_noise_models/__init__.py`; он перенаправляет `__path__`
на `trained_models/sig_noise/`, поэтому старые импорты загружают те же файлы:

```python
# работает по-прежнему
from gplotnikov_sig_noise_models.k_nsol_labelneq0_da_hs128_k0p0001.sig_noise_model_v3 import (
    load_model, predict_flat,
)

# канонический путь для нового кода
from trained_models.sig_noise.k_nsol_labelneq0_da_hs128_k0p0001.sig_noise_model_v3 import (
    load_model, predict_flat,
)
```

`sig_noise_model_v3.py` вычисляет `MODEL_DIR = Path(__file__).parent`, поэтому
`DEFAULT_CHECKPOINT` и `DEFAULT_CONFIG` после переноса указывают на новое место
без правок в коде модели.

Старые **пути к скриптам** (`gplotnikov_sig_noise_models/.../predict_mc_h5.py`)
не сохраняются: используйте `inference/sig_noise/predict_{mc,exp}_h5.py` или
обёртки `run_predict_{mc,exp}.sh`.

## Что не менялось

- веса, `train_config_mc_2020.yaml`, нормализация;
- имена выходных файлов: `{stem}_probs_k_nsol_labelneq0_da_hs128_k0p0001.h5`
  рядом с исходным HDF5;
- схема HDF5 и атрибуты `sig_noise_batch_size` / `sig_noise_model`;
- правило «batch size 256 — часть отбора хитов, а не скорость»
  (`doc/sig_noise_batch_size.md`).

## Проверки (verified, 2026-11-08)

- Старый и новый import path загружают один и тот же файл; `DEFAULT_CHECKPOINT`
  существует.
- Детерминированный fingerprint на 512 событиях / 32 071 хите (CPU, batch 64)
  до и после переноса совпал побитово: SHA-256
  `003b3713eb3155cbc14ae8e92bc860dcf072e1c15c8cdb237640a7517a37419c`.
- Прогон `predict_mc_h5.py` до и после на одном и том же малом HDF5 дал
  побитово одинаковые датасеты и атрибуты.
- `load_sn_model()`, импорты `data_manager/*_ds_builder`, `excess.signal` и
  `inference.nu_classifier.predict_*` работают с новым путём.

## Где смотреть дальше

- `AGENTS.md` — карта проекта и место `trained_models/sig_noise/` в дереве;
- `inference/sig_noise/README.md` — запуск и формат выходных файлов;
- `doc/sig_noise_batch_size.md` — почему batch size определяет отбор;
- `doc/AUDIT.md` — историческая карта до переноса, не обновляется.
