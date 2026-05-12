# Parking Slot Detection

Проект для детекции парковочных мест на surround-view / BEV изображениях и определения статуса занятости места.

Текущая идея пайплайна:

```text
кадр / видео
  -> детектор парковочных мест
  -> кроп каждого места
  -> EfficientNet-B0 классификатор free / occupied
  -> визуализация и JSON-результат
```

## Текущее состояние

Сейчас в репозитории есть рабочий MVP:

- базовый video pipeline в `src/main.py`;
- YOLO vehicle detector, но для surround-view он оказался слабым и не является основным решением;
- адаптер pretrained CRPS-D parking-slot detector;
- EfficientNet-B0 occupancy classifier;
- скрипты для подготовки датасетов, обучения occupancy classifier и оценки результатов;
- конвертер ParkRecon3D BEV в CRPS-D-like формат для будущего дообучения детектора парковочных мест.

Главный вывод по экспериментам: классификатор занятости работает хорошо, а основная проблема сейчас в переносе детектора парковочных мест на другой BEV-домен.

## Примеры

### CRPS-D, полный пайплайн

![CRPS-D full pipeline](docs/assets/crpsd_full_pipeline_example.jpg)

### CRPS-D, проверка occupancy classifier на GT-слотах

![CRPS-D occupancy](docs/assets/crpsd_occupancy_example.jpg)

### ParkRecon3D BEV, текущий перенос CRPS-D detector

![ParkRecon3D BEV](docs/assets/parkrecon3d_bev_example.jpg)

На ParkRecon3D BEV видно, что домен отличается от CRPS-D: текущий detector находит мало слотов, поэтому следующий важный этап - fine-tune slot detector на ParkRecon3D BEV.

## Структура проекта

```text
configs/
  default.yaml                         # основной конфиг пайплайна

src/
  main.py                              # запуск video pipeline
  detection/
    parking_slot_detector.py           # mock / CRPS-D slot detector
    vehicle_detector.py                # YOLO vehicle detector
    schemas.py                         # dataclass-схемы
  occupancy/
    classifier.py                      # EfficientNet-B0 classifier
    estimator.py                       # выбор geometry / classifier backend
    state_manager.py                   # temporal smoothing для видео
  datasets/
    crpsd.py                           # парсер CRPS-D labels
  visualization/
    draw.py                            # отрисовка результатов

scripts/
  train_occupancy_efficientnet.py      # обучение EfficientNet-B0
  visualize_crpsd_occupancy.py         # нарезка occupancy crops из CRPS-D
  evaluate_occupancy_classifier_crpsd.py
  evaluate_full_pipeline_crpsd.py
  evaluate_parkrecon3d_bev.py
  convert_parkrecon3d_bev.py
  test_parking_slot_detector_crpsd.py
  test_vehicle_detector_crpsd.py

docs/assets/
  *.jpg                                # примеры работы для README
```

## Что не хранится в git

Большие файлы игнорируются:

- `models/**/*.pt`
- `outputs/`
- `external/`
- `data/raw/`
- `data/processed/`
- `data/samples/*.mp4`

Их нужно положить локально руками.

Сейчас использовались такие веса:

```text
models/vehicle/yolo11n.pt
models/occupancy/efficientnet_b0_crpsd.pt
/home/slomauh/pretrain_model/pretrain_model/1:2.pth
```

Для CRPS-D detector также нужен внешний репозиторий:

```text
external/CRPS-D
```

Он соответствует проекту:

```text
https://github.com/zzh362/CRPS-D
```

## Установка локально

Рекомендуется Python 3.10+.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Если нужен CRPS-D detector:

```bash
mkdir -p external
git clone https://github.com/zzh362/CRPS-D external/CRPS-D
```

Потом положить веса:

```text
models/occupancy/efficientnet_b0_crpsd.pt
models/vehicle/yolo11n.pt
```

И указать путь к slot detector weights в `configs/default.yaml`:

```yaml
parking_slot_detector:
  backend: "crpsd"
  model_path: "/path/to/1:2.pth"
```

Сейчас в конфиге slot detector по умолчанию стоит `mock`, чтобы базовый pipeline мог запуститься без внешних весов.

## Запуск demo pipeline

```bash
python src/main.py --config configs/default.yaml
```

Если `data/samples/demo.mp4` отсутствует, код сам создаст простой synthetic demo video.

Результаты:

```text
outputs/videos/demo_result.mp4
outputs/json/demo_result.json
```

## Occupancy classifier

Для статуса занятости используется EfficientNet-B0:

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

### Проверка occupancy classifier на CRPS-D

Оценка на GT-слотах CRPS-D test:

```bash
python scripts/evaluate_occupancy_classifier_crpsd.py \
  --dataset-root /home/slomauh/CRPS-D/CRPS-D \
  --split test \
  --device cpu \
  --model-path models/occupancy/efficientnet_b0_crpsd.pt \
  --output-dir outputs/crpsd_occupancy_eval_test_full
```

Полученный результат:

```text
images: 4376
slots: 12103
accuracy: 98.22%
occupied_f1: 98.53%
free_f1: 97.74%
```

Вывод: классификатор занятости на CRPS-D работает хорошо.

## Полная связка на CRPS-D

Оценка:

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

Вывод: occupancy classifier остается сильным на найденных слотах, но end-to-end ограничен качеством slot detector.

## ParkRecon3D BEV

ParkRecon3D был скачан локально сюда:

```text
/home/slomauh/Downloads/data1
```

Для тестов брались только BEV-изображения:

```text
/home/slomauh/Downloads/data1/BEV/Data/Image
/home/slomauh/Downloads/data1/BEV/Data/label
```

В `label/*.json` есть геометрия слотов:

```json
{
  "marks": [[x0, y0, x1, y1, type], ...],
  "slots": [[mark_a, mark_b, slot_type, angle], ...]
}
```

Важное ограничение: в этих labels нет ground truth статуса занятости. Поэтому на ParkRecon3D BEV можно честно оценивать slot detector, но нельзя честно считать accuracy occupancy classifier.

### Текущая проверка на ParkRecon3D BEV

```bash
python scripts/evaluate_parkrecon3d_bev.py \
  --dataset-root /home/slomauh/Downloads/data1 \
  --limit 30 \
  --device cpu \
  --slot-model-path /home/slomauh/pretrain_model/pretrain_model/1:2.pth \
  --occupancy-model-path models/occupancy/efficientnet_b0_crpsd.pt \
  --slot-conf 0.01 \
  --detector-input-size 512 \
  --match-iou 0.10 \
  --output-dir outputs/parkrecon3d_bev_pipeline_test_resize512_conf001
```

Результат на 30 кадрах:

```text
gt_slots: 118
pred_slots: 33
matched_slots: 21
slot_recall: 17.80%
slot_precision: 63.64%
```

Вывод: CRPS-D detector плохо переносится на ParkRecon3D BEV без дообучения.

## Конвертация ParkRecon3D BEV для дообучения detector

Скрипт:

```text
scripts/convert_parkrecon3d_bev.py
```

Команда:

```bash
python scripts/convert_parkrecon3d_bev.py \
  --dataset-root /home/slomauh/Downloads/data1 \
  --output-dir outputs/parkrecon3d_bev_crpsd_format \
  --val-ratio 0.2 \
  --image-size 512 \
  --split-strategy chronological \
  --gap-size 30
```

Что делает:

- берет BEV images и labels;
- ресайзит изображения до `512x512`;
- пересчитывает координаты marks;
- создает raw CRPS-D-like формат для оценки;
- создает prepared формат для train-кода из `external/CRPS-D`;
- делает безопасный хронологический split, а не random split.

Выход:

```text
outputs/parkrecon3d_bev_crpsd_format/
  raw/
    train/img/
    train/slot_label/
    test/img/
    test/slot_label/
  prepared/
    train/
    test/
  summary.json
```

Текущий converted dataset:

```text
total_pairs: 1426
train images: 1111
train slots: 5333
test images: 285
test slots: 1264
dropped gap frames: 30
```

Проверка на утечки:

```text
test with train neighbor <= 1 frame: 0/285
test with train neighbor <= 10 frames: 0/285
test with train neighbor <= 30 frames: 0/285
min nearest frame distance: 31
dHash near duplicates up to 32/256 bits: 0
```

Архив для загрузки в Kaggle:

```text
outputs/kaggle_parkrecon3d_bev_dataset/parkrecon3d_bev_crpsd_format.zip
```

Размер примерно `346M`.

## Что сделано

1. Собран базовый pipeline для видео.
2. Подключен YOLO vehicle detector.
3. Проверено, что vehicle detector плохо подходит для surround-view кадров.
4. Подключен pretrained CRPS-D slot detector.
5. Подготовлены CRPS-D occupancy crops для обучения классификатора.
6. Обучен EfficientNet-B0 occupancy classifier.
7. Встроен classifier backend в occupancy estimation.
8. Проверен classifier на полных CRPS-D кадрах с GT-слотами.
9. Проверена полная связка на CRPS-D.
10. Проверен перенос на ParkRecon3D BEV.
11. Сделан конвертер ParkRecon3D BEV в CRPS-D-like формат.
12. Исправлен split ParkRecon3D BEV: random split заменен на хронологический split с gap, чтобы убрать leakage между train и test.
13. Собран Kaggle zip для ParkRecon3D BEV fine-tune.

## Что планируется дальше

Ближайший важный этап:

1. Дообучить slot detector на `outputs/parkrecon3d_bev_crpsd_format/prepared/train`.
2. Проверить новые веса на `prepared/test` или raw test через `evaluate_parkrecon3d_bev.py`.
3. Сравнить с текущим baseline:
   - baseline recall на ParkRecon3D BEV: `17.8%`;
   - цель после fine-tune: существенно поднять recall без сильной просадки precision.

После этого:

4. Подключить новые slot detector weights в `configs/default.yaml`.
5. Перепроверить полную связку detector + occupancy classifier.
6. Если будет датасет с occupancy labels для BEV, дообучить occupancy classifier уже под ParkRecon3D-like домен.
7. Сделать удобный inference script для папки кадров и для видео.
8. Добавить нормальный train/fine-tune script для CRPS-D detector, чтобы не править `external/CRPS-D/train.py` руками.

## Важные замечания

- CRPS-D detector обучался на `512x512`, поэтому для ParkRecon3D BEV обязательно нужен resize или отдельное обучение под исходное разрешение.
- ParkRecon3D BEV является раскадровкой видео, поэтому random split дает leakage. Использовать только chronological split с gap.
- В текущем ParkRecon3D BEV labels нет occupancy GT, значит нельзя честно оценить free/occupied accuracy на этом датасете.
- Большие веса и датасеты не лежат в git, их нужно передавать отдельно или загружать через Kaggle datasets.
