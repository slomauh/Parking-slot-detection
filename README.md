# Parking Slot Detection

Проект для детекции парковочных мест на surround-view / BEV изображениях и определения статуса занятости места.

Текущий рабочий пайплайн:

```text
кадр / последовательность кадров
  -> detector парковочных мест
  -> crop каждого найденного места
  -> EfficientNet-B0 classifier free / occupied
  -> preview, JSON, галереи crops, видео
```

## Текущее состояние

Главный вывод после последних экспериментов: CRPS-D marking-point detector оказался ограничен pairing/postprocess логикой, а YOLO-OBB на ParkRecon3D BEV резко поднял качество. Поэтому основной кандидат на slot detector сейчас - YOLO-OBB, который сразу предсказывает повернутый четырехугольник парковочного места.

В репозитории есть:

- базовый video pipeline в `src/main.py`;
- CRPS-D slot detector backend как baseline/fallback;
- YOLO-OBB slot detector backend;
- EfficientNet-B0 occupancy classifier;
- конвертеры ParkRecon3D BEV в CRPS-D-like и YOLO-OBB форматы;
- evaluation/QA скрипты для CRPS-D, ParkRecon3D BEV, temporal sequence и полного pipeline;
- Kaggle-инструкции для обучения CRPS-D fine-tune и YOLO-OBB.

Текущий лучший практический путь:

```text
ParkRecon3D BEV
  -> YOLO-OBB slot detector, conf около 0.55
  -> EfficientNet-B0 occupancy classifier
  -> visual QA / JSON / video
```

## Примеры

### CRPS-D, полный пайплайн

![CRPS-D full pipeline](docs/assets/crpsd_full_pipeline_example.jpg)

### CRPS-D, проверка occupancy classifier на GT-слотах

![CRPS-D occupancy](docs/assets/crpsd_occupancy_example.jpg)

### ParkRecon3D BEV

![ParkRecon3D BEV](docs/assets/parkrecon3d_bev_example.jpg)

## Структура проекта

```text
configs/
  default.yaml

src/
  main.py
  detection/
    parking_slot_detector.py       # mock / CRPS-D / YOLO-OBB slot detector
    vehicle_detector.py            # YOLO vehicle detector
    schemas.py
  occupancy/
    classifier.py                  # EfficientNet-B0 classifier
    estimator.py
    state_manager.py
  datasets/
    crpsd.py
  visualization/
    draw.py

scripts/
  train_occupancy_efficientnet.py
  evaluate_occupancy_classifier_crpsd.py
  evaluate_full_pipeline_crpsd.py
  evaluate_parkrecon3d_bev.py
  evaluate_parkrecon3d_temporal_slots.py
  convert_parkrecon3d_bev.py
  convert_parkrecon3d_yolo_obb.py
  prepare_parkrecon3d_camera_images.py
  visualize_parkrecon3d_resized_labels.py

docs/
  kaggle_yolo_obb.md
  kaggle_slot_detector_finetune.md
  parkrecon3d_camera_projection.md
  assets/
```

## Что не хранится в git

Большие файлы игнорируются:

```text
models/**/*.pt
models/**/*.pth
outputs/
external/
data/raw/
data/processed/
data/samples/*.mp4
```

Локально сейчас использовались такие веса:

```text
models/vehicle/yolo11n.pt
models/occupancy/efficientnet_b0_crpsd.pt
models/slot_detector/best_yolo_parkrecon.pt
models/slot_detector/parkrecon3d_slot_detector_finetuned.pth
```

Для CRPS-D backend нужен внешний репозиторий:

```text
external/CRPS-D
https://github.com/zzh362/CRPS-D
```

## Установка локально

Рекомендуется Python 3.10+.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Если нужен CRPS-D backend:

```bash
mkdir -p external
git clone https://github.com/zzh362/CRPS-D external/CRPS-D
```

Для YOLO-OBB нужен пакет `ultralytics`. Если его нет после установки requirements:

```bash
pip install ultralytics
```

## Конфиг

`configs/default.yaml` пока хранит CRPS-D/postprocess настройки как совместимый baseline. Для YOLO-OBB нужно переключить блок `parking_slot_detector` так:

```yaml
parking_slot_detector:
  backend: "yolo_obb"
  model_path: "models/slot_detector/best_yolo_parkrecon.pt"
  device: "cuda"
  conf_threshold: 0.55
  imgsz: 1024
```

Если CUDA недоступна, использовать `device: "cpu"`.

## Запуск demo pipeline

```bash
python src/main.py --config configs/default.yaml
```

Если `data/samples/demo.mp4` отсутствует, код создает synthetic demo video.

Результаты:

```text
outputs/videos/demo_result.mp4
outputs/json/demo_result.json
```

## Occupancy classifier

Статус занятости считает EfficientNet-B0:

```text
src/occupancy/classifier.py
```

Классы:

```text
free
occupied
```

Ожидаемый путь весов:

```text
models/occupancy/efficientnet_b0_crpsd.pt
```

Проверка на GT-слотах CRPS-D test:

```bash
python scripts/evaluate_occupancy_classifier_crpsd.py \
  --dataset-root /home/slomauh/CRPS-D/CRPS-D \
  --split test \
  --device cpu \
  --model-path models/occupancy/efficientnet_b0_crpsd.pt \
  --output-dir outputs/crpsd_occupancy_eval_test_full
```

Результат:

```text
images: 4376
slots: 12103
accuracy: 98.22%
occupied_f1: 98.53%
free_f1: 97.74%
```

Вывод: classifier на CRPS-D работает хорошо. На ParkRecon3D BEV честной occupancy accuracy пока нет, потому что в датасете нет GT `free/occupied`.

## CRPS-D full pipeline

Оценка полной связки на CRPS-D:

```bash
python scripts/evaluate_full_pipeline_crpsd.py \
  --dataset-root /home/slomauh/CRPS-D/CRPS-D \
  --split test \
  --device cpu \
  --slot-model-path /home/slomauh/pretrain_model/pretrain_model/1:2.pth \
  --occupancy-model-path models/occupancy/efficientnet_b0_crpsd.pt \
  --match-iou 0.10 \
  --output-dir outputs/crpsd_full_pipeline_eval_test_full_iou010
```

Результат:

```text
images: 4376
gt_slots: 12103
pred_slots: 11799
matched_slots: 10130

slot_recall: 83.70%
slot_precision: 85.85%
matched_occupancy_accuracy: 98.00%
end_to_end_slot_status_accuracy_over_gt: 82.02%
```

## ParkRecon3D BEV dataset

ParkRecon3D был подготовлен из локальных частей:

```text
/home/slomauh/Documents/data1
/home/slomauh/Documents/data2
/home/slomauh/Documents/data3
```

Используются BEV-изображения:

```text
<dataset_part>/BEV/Data/Image
<dataset_part>/BEV/Data/label
```

В labels есть геометрия парковочных мест:

```json
{
  "marks": [[x0, y0, x1, y1, type]],
  "slots": [[mark_a, mark_b, slot_type, angle]]
}
```

Важное ограничение: в ParkRecon3D labels нет статуса занятости.

Конвертация в CRPS-D-like формат:

```bash
python scripts/convert_parkrecon3d_bev.py \
  --dataset-roots \
    /home/slomauh/Documents/data1 \
    /home/slomauh/Documents/data2 \
    /home/slomauh/Documents/data3 \
  --output-dir outputs/parkrecon3d_bev_crpsd_format \
  --val-ratio 0.2 \
  --image-size 512 \
  --split-strategy chronological \
  --gap-size 30
```

Текущий split:

```text
total_pairs: 5005
duplicate_pairs_dropped: 429
train images: 3974
train slots: 15937
test images: 1001
test slots: 4786
dropped gap frames: 30
```

Проверка leakage:

```text
test with train neighbor <= 1 frame: 0/1001
test with train neighbor <= 10 frames: 0/1001
test with train neighbor <= 30 frames: 0/1001
min nearest frame distance: 31
dHash exact duplicates: 0
dHash near duplicates up to 8/256 bits: 0
```

## CRPS-D detector на ParkRecon3D

Изначальный pretrained CRPS-D detector без дообучения плохо переносился на ParkRecon3D BEV:

```text
30-frame smoke:
gt_slots: 118
pred_slots: 33
matched_slots: 21
slot_recall: 17.80%
slot_precision: 63.64%
```

После fine-tune и большого числа postprocess экспериментов лучший CRPS-D-like вариант уперся примерно в:

```text
recall:    77.48%
precision: 77.54%
F1:        77.51%
```

Основная проблема: модель предсказывает marking points, а pairing/postprocess иногда собирает поперечные ложные слоты или дубли. Поэтому CRPS-D backend оставлен как baseline/fallback, но не выглядит лучшим путем дальше.

## YOLO-OBB detector на ParkRecon3D

Новый эксперимент: обучать прямой oriented-box detector, где каждый слот - четырехугольник.

Конвертация датасета:

```bash
python scripts/convert_parkrecon3d_yolo_obb.py \
  --source-dir outputs/parkrecon3d_bev_crpsd_format/raw \
  --output-dir outputs/parkrecon3d_yolo_obb
```

Kaggle zip:

```text
outputs/kaggle_parkrecon3d_yolo_obb/parkrecon3d_yolo_obb.zip
```

Инструкция:

```text
docs/kaggle_yolo_obb.md
```

Локальная проверка YOLO-OBB:

```bash
python scripts/evaluate_parkrecon3d_bev.py \
  --image-dir outputs/parkrecon3d_bev_crpsd_format/raw/test/img \
  --label-dir outputs/parkrecon3d_bev_crpsd_format/raw/test/slot_label \
  --slot-backend yolo_obb \
  --slot-model-path models/slot_detector/best_yolo_parkrecon.pt \
  --slot-conf 0.55 \
  --detector-input-size 1024 \
  --device cpu \
  --skip-occupancy \
  --preview-limit 100 \
  --output-dir outputs/parkrecon3d_yolo_obb_eval_conf055_full_test
```

Текущий checkpoint после 15 эпох на Kaggle:

```text
slot_conf: 0.55
images: 1001
gt_slots: 4786
pred_slots: 4952
matched_slots: 4574
false_negative_slots: 212
false_positive_slots: 378

recall:    95.57%
precision: 92.37%
F1:        93.94%
```

Sweep по `slot_conf` показал:

```text
0.25: recall 98.47%, precision 88.89%, F1 93.44%
0.55: recall 95.84%, precision 93.57%, F1 94.69%  # лучший F1 в single-pass sweep
0.60: recall 95.13%, precision 94.15%, F1 94.64%  # чуть чище, почти тот же F1
0.70: recall 93.36%, precision 95.47%, F1 94.40%
```

Практический вывод: `slot_conf=0.55` - текущий лучший баланс. Если нужно меньше визуального мусора, можно пробовать `0.60`.

## Full pipeline на ParkRecon3D BEV

Для визуальной проверки detector + occupancy classifier:

```bash
python scripts/evaluate_parkrecon3d_bev.py \
  --image-dir outputs/parkrecon3d_bev_crpsd_format/raw/test/img \
  --label-dir outputs/parkrecon3d_bev_crpsd_format/raw/test/slot_label \
  --slot-backend yolo_obb \
  --slot-model-path models/slot_detector/best_yolo_parkrecon.pt \
  --slot-conf 0.55 \
  --detector-input-size 1024 \
  --occupancy-model-path models/occupancy/efficientnet_b0_crpsd.pt \
  --device cpu \
  --preview-limit 100 \
  --output-dir outputs/parkrecon3d_yolo_obb_occupancy_qa_conf055
```

Выход:

```text
summary.json
predictions.json
preview/
crops/free/
crops/occupied/
crops/low_confidence/
contact_sheet.jpg
```

Так как occupancy GT в ParkRecon3D нет, это именно visual QA, а не честная accuracy.

## ParkRecon3D Camera0/Camera1/Camera2

В ParkRecon3D есть обычные камеры:

```text
<dataset_part>/Camera0/Data/Image
<dataset_part>/Camera1/Data/Image
<dataset_part>/Camera2/Data/Image
```

В этих папках нет `label/*.json` с разметкой парковочных мест. Они подготовлены как image-only splits для визуальной проверки, pseudo-labeling или будущей ручной разметки:

```bash
python scripts/prepare_parkrecon3d_camera_images.py \
  --dataset-roots \
    /home/slomauh/Documents/data1 \
    /home/slomauh/Documents/data2 \
    /home/slomauh/Documents/data3 \
  --cameras Camera0 Camera1 Camera2 \
  --output-dir outputs/parkrecon3d_camera_images \
  --image-size 512 \
  --val-ratio 0.2 \
  --split-strategy chronological \
  --gap-size 30
```

`IMU` и `Wheel` содержат CSV с сенсорикой автомобиля: ускорения, угловые скорости, колесная одометрия, скорость/поворот. Для текущего image-only detector они не используются, но могут пригодиться для ego-motion, локализации, стабилизации sequence и 3D-реконструкции.

Эксперимент с проекцией BEV labels на камеры описан здесь:

```text
docs/parkrecon3d_camera_projection.md
```

Короткий вывод: проекция возможна, но без сильной фильтрации видимости разметка получается грязной.

## Что сделано

1. Собран базовый pipeline для видео.
2. Подключен YOLO vehicle detector; для surround-view он оказался не основным решением.
3. Подключен pretrained CRPS-D slot detector.
4. Подготовлены CRPS-D occupancy crops.
5. Обучен и встроен EfficientNet-B0 occupancy classifier.
6. Проверена occupancy accuracy на CRPS-D GT slots.
7. Проверена полная связка на CRPS-D.
8. Подготовлен ParkRecon3D BEV split с chronological gap и проверкой leakage.
9. Дообучен CRPS-D-like slot detector на ParkRecon3D BEV.
10. Проведены эксперименты с prepared conversion, relaxed pairing, row-consensus postprocess и temporal smoothing.
11. Сделан вывод, что CRPS-D pairing близок к потолку для ParkRecon3D.
12. Подготовлен YOLO-OBB датасет.
13. Обучен YOLO-OBB checkpoint на 15 эпохах.
14. YOLO-OBB проверен на полном ParkRecon3D test split и стал основным кандидатом.

## Что делать дальше

Ближайшие шаги:

1. Дождаться полного YOLO-OBB обучения на 80 эпохах.
2. Положить новый `best.pt` в `models/slot_detector/`.
3. Повторить `slot_conf` sweep, потому что оптимальный порог может сместиться.
4. Переключить `configs/default.yaml` на YOLO-OBB как основной backend, если новый checkpoint подтверждает качество.
5. Прогнать full pipeline `YOLO-OBB -> occupancy classifier` на полном ParkRecon3D test sequence.
6. Собрать видео/sequence preview.
7. Если occupancy визуально ошибается на ParkRecon3D, собрать 300-1000 crops и вручную доразметить `free/occupied` для дообучения classifier под новый домен.

## Важные замечания

- ParkRecon3D BEV является раскадровкой видео, поэтому использовать только chronological split с gap.
- В ParkRecon3D BEV нет occupancy GT, значит `free/occupied` на этом датасете проверяется визуально.
- CRPS-D backend не удален, но новый основной путь - YOLO-OBB.
- Большие веса и датасеты не лежат в git, их нужно передавать отдельно или загружать через Kaggle datasets.
