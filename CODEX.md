# CODEX.md — контекст проекта для Codex

Дата: 2026-04-29  
Проект: автоматическая детекция парковочных мест и определение занятости в реальном времени  
Сценарий: **камера установлена на движущемся автомобиле**, а не стационарная камера на парковке.

---


## Содержание

- [1. Что мы вообще делаем](#1-что-мы-вообще-делаем)
- [2. Важное ограничение: это НЕ стационарная камера](#2-важное-ограничение-это-не-стационарная-камера)
- [3. Итоговая архитектура MVP](#3-итоговая-архитектура-mvp)
  - [3.1. Основной pipeline](#31-основной-pipeline)
- [4. Выбранная стратегия реализации](#4-выбранная-стратегия-реализации)
  - [4.1. Не обучать всё с нуля](#41-не-обучать-всё-с-нуля)
  - [4.2. Что можно обучать](#42-что-можно-обучать)
- [5. Датасеты](#5-датасеты)
  - [5.1. CRPS-D](#51-crps-d)
  - [5.2. PS 2.0 / Tongji Parking-slot Dataset 2.0](#52-ps-20-tongji-parking-slot-dataset-20)
  - [5.3. Dashcam Video Dataset из работы Wu & Yeh](#53-dashcam-video-dataset-из-работы-wu-yeh)
  - [5.4. DLP / Dragon Lake Parking Dataset](#54-dlp-dragon-lake-parking-dataset)
  - [5.5. ParkingScenes / CARLA simulation](#55-parkingscenes-carla-simulation)
- [6. Модели и алгоритмы](#6-модели-и-алгоритмы)
  - [6.1. Детектор парковочных мест](#61-детектор-парковочных-мест)
  - [6.2. Детектор машин и людей](#62-детектор-машин-и-людей)
  - [6.3. Трекинг](#63-трекинг)
- [7. Определение занятости](#7-определение-занятости)
  - [7.1. Базовая логика](#71-базовая-логика)
  - [7.2. IoU / overlap](#72-iou-overlap)
  - [7.3. Point-in-Polygon](#73-point-in-polygon)
  - [7.4. Учет скорости](#74-учет-скорости)
  - [7.5. Счетчики кадров](#75-счетчики-кадров)
- [8. Предсказание / дополнительная новизна](#8-предсказание-дополнительная-новизна)
  - [8.1. Статус "скоро освободится"](#81-статус-скоро-освободится)
  - [8.2. Статус "потенциально занято"](#82-статус-потенциально-занято)
  - [8.3. Вероятность освобождения по истории](#83-вероятность-освобождения-по-истории)
- [9. Реальное время](#9-реальное-время)
- [10. Ограничения по вычислениям](#10-ограничения-по-вычислениям)
  - [10.1. Что реально потянет Colab Free / Kaggle](#101-что-реально-потянет-colab-free-kaggle)
  - [10.2. Рекомендуемая стратегия](#102-рекомендуемая-стратегия)
  - [10.3. Настройки для экономии GPU](#103-настройки-для-экономии-gpu)
- [11. Предлагаемая структура проекта](#11-предлагаемая-структура-проекта)
- [12. Минимальный `requirements.txt`](#12-минимальный-requirementstxt)
- [13. Конфиг MVP](#13-конфиг-mvp)
- [14. Основной псевдокод](#14-основной-псевдокод)
- [15. Минимальная последовательность задач для Codex](#15-минимальная-последовательность-задач-для-codex)
  - [Этап A — каркас проекта](#этап-a-каркас-проекта)
  - [Этап B — YOLO vehicle detection](#этап-b-yolo-vehicle-detection)
  - [Этап C — tracking](#этап-c-tracking)
  - [Этап D — parking slot detector](#этап-d-parking-slot-detector)
  - [Этап E — occupancy logic](#этап-e-occupancy-logic)
  - [Этап F — output](#этап-f-output)
- [16. Что считать успешным MVP](#16-что-считать-успешным-mvp)
- [17. Что НЕ делать в первой версии](#17-что-не-делать-в-первой-версии)
- [18. Эксперименты для отчета/диплома](#18-эксперименты-для-отчетадиплома)
- [19. Метрики](#19-метрики)
- [20. Формулировка исследовательской идеи](#20-формулировка-исследовательской-идеи)
- [21. Короткое описание для README](#21-короткое-описание-для-readme)
- [22. Источники и полезные ссылки](#22-источники-и-полезные-ссылки)
  - [Датасеты](#датасеты)
  - [Инструменты](#инструменты)
  - [Вычислительные ресурсы](#вычислительные-ресурсы)
- [23. Приоритеты для Codex](#23-приоритеты-для-codex)
- [24. Главная мысль](#24-главная-мысль)

---

## 1. Что мы вообще делаем

Нужно реализовать прототип системы, которая по видео с автомобиля:

1. получает кадры из видео/камеры;
2. находит парковочные места в текущем кадре;
3. находит автомобили, людей и другие релевантные объекты;
4. определяет, свободно парковочное место или занято;
5. стабилизирует решение во времени через трекинг и счетчики кадров;
6. выводит результат поверх видео и/или сохраняет JSON с состоянием мест.

Главная идея: **не фиксированная разметка парковки**, а покадровая детекция парковочных мест, потому что камера движется и перспектива постоянно меняется.

---

## 2. Важное ограничение: это НЕ стационарная камера

Нельзя строить решение так:

```text
один раз вручную размечаем полигоны всех парковочных мест
→ потом постоянно проверяем эти фиксированные полигоны
```

Такой подход годится для камер на столбах/зданиях, но не подходит для камеры на машине.

Для нашего случая нужно:

```text
каждый кадр / каждый N-й кадр
→ заново детектировать parking slots
→ сопоставлять их с машинами
→ трекать места и машины между кадрами
→ обновлять статус места
```

---

## 3. Итоговая архитектура MVP

```text
video_source
    ↓
FrameReader / OpenCV VideoCapture
    ↓
ParkingSlotDetector
    ↓
VehicleDetector
    ↓
Tracker
    ↓
OccupancyEstimator
    ↓
TemporalStateManager
    ↓
Visualizer + JSON Export
```

### 3.1. Основной pipeline

```text
1. Считать кадр из видео.
2. На каждом N-м кадре запустить детектор парковочных мест.
3. На каждом N-м кадре или чаще запустить YOLO для машин/людей.
4. Присвоить объектам track_id через ByteTrack/BoT-SORT.
5. Для каждого найденного parking slot проверить:
   - пересекается ли слот с bbox машины;
   - находится ли нижний центр машины внутри полигона/зоны слота;
   - насколько машина движется быстро или медленно.
6. Не менять статус по одному кадру:
   - статус "занято" только после нескольких подтверждений подряд;
   - статус "свободно" только после нескольких кадров отсутствия машины.
7. Нарисовать результат на кадре:
   - зеленый — свободно;
   - красный — занято;
   - желтый — потенциально занято / скоро освободится.
8. Сохранить результат в JSON.
```

---

## 4. Выбранная стратегия реализации

### 4.1. Не обучать всё с нуля

У нас нет кластера. Есть только:

- Colab Free;
- Kaggle;
- возможно локальный ноутбук с RTX 3050 Laptop.

Поэтому нельзя делать ставку на тяжелое обучение с нуля. Основная стратегия:

```text
готовый / предобученный parking-slot detector
+ готовый YOLO для машин
+ трекинг
+ собственная логика связывания и стабилизации
```

### 4.2. Что можно обучать

Разрешено:

- дообучение легкой модели на части данных;
- эксперименты на subset датасета;
- обучение только head при замороженном backbone;
- small/nano модели;
- небольшие эпохи.

Не делать основой проекта:

- обучение большой модели с нуля;
- semi-supervised teacher-student baseline на всём CRPS-D;
- обучение нескольких больших моделей одновременно;
- тяжелую segmentation-модель, если detector достаточно.

---

## 5. Датасеты

### 5.1. CRPS-D

Использовать как **основной датасет для детекции парковочных мест**.

Почему:

- крупный датасет для parking slot detection;
- разные условия освещения;
- разная погода;
- сложные варианты парковочных мест;
- есть репозиторий с кодом и pretrained model;
- подходит для обучения/проверки детектора парковочных мест.

Ограничение:

- это в основном покадровая детекция;
- не полноценный видеодатасет для трекинга;
- не закрывает всю задачу real-time pipeline сам по себе.

Роль в проекте:

```text
CRPS-D → parking slot detector
```

---

### 5.2. PS 2.0 / Tongji Parking-slot Dataset 2.0

Использовать как **дополнительный benchmark**.

Почему:

- классический датасет для parking slot detection;
- surround-view изображения с автомобиля;
- есть разные типы мест: perpendicular / parallel / slanted;
- полезен для сравнения и отчета.

Ограничение:

- изображения, а не полноценное видео;
- не решает задачу временной логики;
- может быть проще и менее разнообразен, чем CRPS-D.

Роль в проекте:

```text
PS 2.0 → сравнение / baseline / дополнительная проверка
```

---

### 5.3. Dashcam Video Dataset из работы Wu & Yeh

Использовать как **самый близкий открытый видеодатасет для сценария камеры из машины**.

Почему:

- это именно dashcam-видео;
- видео сняты с точки зрения водителя;
- задача связана с ранним обнаружением свободных мест;
- подходит для проверки идеи на видеопоследовательностях.

Ограничение:

- больше про наличие свободного места, чем про точную геометрию parking slot;
- может не содержать нужной разметки полигонов/углов каждого места;
- не заменяет CRPS-D/PS2.0 для обучения slot detector.

Роль в проекте:

```text
Dashcam dataset → проверка real-time/video pipeline
```

---

### 5.4. DLP / Dragon Lake Parking Dataset

Использовать только как вспомогательный датасет.

Почему:

- много видео на парковке;
- есть траектории автомобилей, пешеходов, велосипедистов;
- полезен для анализа поведения и трекинга.

Ограничение:

- камера не с машины, а сверху с дрона;
- не подходит напрямую для визуального восприятия водителя.

Роль в проекте:

```text
DLP → трекинг, поведение, траектории, логика движения
```

---

### 5.5. ParkingScenes / CARLA simulation

Использовать опционально.

Почему:

- есть движение автомобиля;
- можно получить последовательности;
- симуляция может помочь для демонстрации pipeline.

Ограничение:

- это синтетика;
- есть sim-to-real gap;
- не стоит делать основным источником доказательства качества.

Роль в проекте:

```text
ParkingScenes → опциональная симуляционная проверка
```

---

## 6. Модели и алгоритмы

### 6.1. Детектор парковочных мест

Варианты:

1. использовать pretrained model из CRPS-D;
2. дообучить легкую модель на части CRPS-D;
3. отдельно проверить на PS 2.0.

Выход детектора парковочных мест должен быть приведен к единому формату:

```python
ParkingSlot = {
    "slot_id": int,
    "points": [[x1, y1], [x2, y2], [x3, y3], [x4, y4]],
    "confidence": float,
    "type": "parallel|perpendicular|slanted|unknown"
}
```

Если модель выдает marking points, надо восстановить полигон/ориентацию слота.

---

### 6.2. Детектор машин и людей

Использовать готовый YOLO, без обучения.

Классы:

```text
car
truck
bus
motorcycle — по необходимости
person — для прогноза освобождения/окклюзий
```

Предпочтительно:

```text
YOLO11n / YOLO11s
или YOLOv8n / YOLOv8s
```

Причина: легкие модели, нормальны для Colab/Kaggle и real-time inference.

Выход привести к формату:

```python
Detection = {
    "class_name": "car",
    "bbox": [x1, y1, x2, y2],
    "confidence": float
}
```

---

### 6.3. Трекинг

Использовать:

```text
ByteTrack
или BoT-SORT
```

Задача трекера:

```text
bbox машины на разных кадрах → один track_id
```

Выход:

```python
Track = {
    "track_id": int,
    "class_name": "car",
    "bbox": [x1, y1, x2, y2],
    "center": [cx, cy],
    "speed_px": float,
    "age": int,
    "missed_frames": int
}
```

Скорость можно сначала считать просто:

```python
speed_px = euclidean_distance(center_t, center_t_minus_1)
```

---

## 7. Определение занятости

### 7.1. Базовая логика

Для каждого найденного parking slot:

1. ищем ближайшую/пересекающуюся машину;
2. считаем overlap между slot polygon и car bbox;
3. проверяем нижний центр bbox машины;
4. учитываем скорость машины;
5. обновляем счетчики.

---

### 7.2. IoU / overlap

Можно использовать:

```text
intersection_area(slot_polygon, car_bbox) / area(slot_polygon)
```

или классический IoU:

```text
intersection_area / union_area
```

Для MVP проще использовать долю покрытия слота машиной:

```python
slot_coverage = intersection_area(slot_polygon, car_bbox_polygon) / area(slot_polygon)
```

Примерное правило:

```python
if slot_coverage > 0.20:
    candidate_occupied = True
```

Порог потом подобрать экспериментально.

---

### 7.3. Point-in-Polygon

Дополнительная проверка:

```python
bottom_center = ((x1 + x2) / 2, y2)
```

Если нижняя центральная точка машины находится внутри полигона parking slot, это сильный признак занятости.

```python
if point_in_polygon(bottom_center, slot_polygon):
    candidate_occupied = True
```

---

### 7.4. Учет скорости

Если машина пересекает слот, но быстро движется, не надо сразу считать место занятым.

Пример:

```python
if candidate_occupied and track.speed_px < SPEED_THRESHOLD:
    occupied_candidate = True
else:
    occupied_candidate = False
```

Примерные начальные параметры:

```python
SPEED_THRESHOLD = 2.0  # пикселя/кадр, потом подбирать
OCCUPIED_CONFIRM_FRAMES = 5
FREE_CONFIRM_FRAMES = 10
```

---

### 7.5. Счетчики кадров

Для каждого слота хранить состояние:

```python
SlotState = {
    "slot_id": int,
    "status": "free|occupied|potentially_occupied|soon_free|unknown",
    "occupied_counter": int,
    "free_counter": int,
    "last_seen_frame": int,
    "assigned_track_id": int | None
}
```

Правило:

```python
if occupied_candidate:
    occupied_counter += 1
    free_counter = 0
else:
    free_counter += 1
    occupied_counter = 0

if occupied_counter >= OCCUPIED_CONFIRM_FRAMES:
    status = "occupied"

if free_counter >= FREE_CONFIRM_FRAMES:
    status = "free"
```

---

## 8. Предсказание / дополнительная новизна

Это не обязательный MVP, но можно добавить как исследовательскую часть.

### 8.1. Статус "скоро освободится"

Идея:

```text
если человек долго находится рядом с припаркованным автомобилем,
особенно у водительской двери,
то можно повысить вероятность освобождения места.
```

Упрощение для MVP:

```text
person bbox рядом с car bbox > 5 секунд
→ status = "soon_free"
```

---

### 8.2. Статус "потенциально занято"

Идея:

```text
если машина медленно движется к свободному месту,
можно временно пометить место как potentially_occupied.
```

Упрощенное правило:

```python
if car_speed_low and distance(car_center, free_slot_center) decreasing:
    status = "potentially_occupied"
```

---

### 8.3. Вероятность освобождения по истории

Если есть история парковки:

```python
release_probability = min(1.0, elapsed_time / mean_parking_duration_for_hour)
```

Для MVP это можно оставить как отдельный модуль-заготовку, без обязательной реализации.

---

## 9. Реальное время

Цель: не максимальный FPS, а стабильная работа.

Достаточно:

```text
5–10 FPS для обработки
задержка до 0.5–1.0 секунды допустима для прототипа
```

Оптимизация:

```text
- обрабатывать каждый N-й кадр;
- запускать тяжелый parking slot detector реже;
- YOLO car detector брать nano/small;
- использовать tracker между детекциями;
- уменьшить разрешение кадра;
- использовать mixed precision, если есть GPU.
```

Пример:

```python
DETECT_EVERY_N_FRAMES = 3
PARKING_SLOT_EVERY_N_FRAMES = 5
```

---

## 10. Ограничения по вычислениям

### 10.1. Что реально потянет Colab Free / Kaggle

Реально:

```text
- inference готовой модели;
- YOLO nano/small inference;
- ByteTrack / BoT-SORT;
- обработка видеофайла;
- дообучение легкой модели на части данных;
- эксперименты с уменьшенным imgsz.
```

Сложно:

```text
- полное обучение большой модели с нуля;
- тяжелая segmentation-модель;
- semi-supervised teacher-student обучение на всем CRPS-D;
- долгие multi-day эксперименты.
```

### 10.2. Рекомендуемая стратегия

```text
Kaggle — основное место для обучения/валидации.
Colab Free — быстрые проверки, inference, демо.
Локальный ноутбук — отладка кода и легкий inference.
```

### 10.3. Настройки для экономии GPU

```yaml
imgsz: 416 или 512
batch: 4, 8 или 16
epochs: 20-50
model_size: nano/small
amp: true
freeze_backbone_first: true
dataset_subset: 3000-8000 изображений
```

---

## 11. Предлагаемая структура проекта

```text
parking-realtime/
│
├── CODEX.md
├── README.md
├── requirements.txt
├── configs/
│   ├── default.yaml
│   ├── tracker_bytetrack.yaml
│   └── thresholds.yaml
│
├── data/
│   ├── raw/
│   ├── processed/
│   └── samples/
│
├── models/
│   ├── parking_slot/
│   └── vehicle/
│
├── src/
│   ├── main.py
│   ├── video/
│   │   ├── reader.py
│   │   └── writer.py
│   │
│   ├── detection/
│   │   ├── parking_slot_detector.py
│   │   ├── vehicle_detector.py
│   │   └── schemas.py
│   │
│   ├── tracking/
│   │   ├── tracker.py
│   │   └── motion.py
│   │
│   ├── geometry/
│   │   ├── polygons.py
│   │   ├── iou.py
│   │   └── point_in_polygon.py
│   │
│   ├── occupancy/
│   │   ├── estimator.py
│   │   ├── state_manager.py
│   │   └── prediction.py
│   │
│   ├── visualization/
│   │   └── draw.py
│   │
│   └── utils/
│       ├── config.py
│       └── logging.py
│
├── notebooks/
│   ├── 01_check_datasets.ipynb
│   ├── 02_slot_detector_inference.ipynb
│   ├── 03_yolo_vehicle_inference.ipynb
│   └── 04_video_pipeline_demo.ipynb
│
├── outputs/
│   ├── videos/
│   ├── json/
│   └── screenshots/
│
└── tests/
    ├── test_geometry.py
    ├── test_occupancy.py
    └── test_state_manager.py
```

---

## 12. Минимальный `requirements.txt`

```text
ultralytics
opencv-python
numpy
torch
torchvision
shapely
pyyaml
tqdm
matplotlib
pandas
```

Если `shapely` будет проблемой в Colab/Kaggle, геометрию можно временно реализовать через OpenCV:

```python
cv2.pointPolygonTest()
cv2.intersectConvexConvex()
```

---

## 13. Конфиг MVP

Создать `configs/default.yaml`:

```yaml
video:
  source: "data/samples/demo.mp4"
  output_path: "outputs/videos/demo_result.mp4"
  save_json: true
  json_path: "outputs/json/demo_result.json"

runtime:
  device: "cuda"
  detect_every_n_frames: 3
  parking_slot_every_n_frames: 5
  max_frames: null

vehicle_detector:
  model_path: "yolo11n.pt"
  classes:
    - car
    - truck
    - bus
    - motorcycle
    - person
  conf_threshold: 0.35
  imgsz: 640

parking_slot_detector:
  model_path: "models/parking_slot/pretrained.pt"
  conf_threshold: 0.30
  imgsz: 512

tracker:
  type: "bytetrack"
  config_path: "configs/tracker_bytetrack.yaml"

occupancy:
  slot_coverage_threshold: 0.20
  speed_threshold_px: 2.0
  occupied_confirm_frames: 5
  free_confirm_frames: 10
  max_missing_slot_frames: 10

visualization:
  draw_tracks: true
  draw_slots: true
  draw_status_text: true
```

---

## 14. Основной псевдокод

```python
def main():
    config = load_config("configs/default.yaml")

    video = VideoReader(config.video.source)
    writer = VideoWriter(config.video.output_path)

    slot_detector = ParkingSlotDetector(config.parking_slot_detector)
    vehicle_detector = VehicleDetector(config.vehicle_detector)
    tracker = Tracker(config.tracker)

    occupancy = OccupancyEstimator(config.occupancy)
    state_manager = TemporalStateManager(config.occupancy)

    last_slots = []

    for frame_id, frame in video:
        if frame_id % config.runtime.parking_slot_every_n_frames == 0:
            slots = slot_detector.predict(frame)
            last_slots = slots
        else:
            slots = last_slots

        if frame_id % config.runtime.detect_every_n_frames == 0:
            detections = vehicle_detector.predict(frame)
            tracks = tracker.update(detections, frame)
        else:
            tracks = tracker.predict(frame)

        candidate_states = occupancy.estimate(
            slots=slots,
            tracks=tracks,
            frame_id=frame_id
        )

        stable_states = state_manager.update(candidate_states, frame_id)

        result_frame = draw_result(
            frame=frame,
            slots=slots,
            tracks=tracks,
            states=stable_states
        )

        writer.write(result_frame)
        save_json_if_needed(stable_states, frame_id)

    video.release()
    writer.release()
```

---

## 15. Минимальная последовательность задач для Codex

### Этап A — каркас проекта

1. Создать структуру папок.
2. Создать `requirements.txt`.
3. Создать `configs/default.yaml`.
4. Создать заглушки классов:
   - `VideoReader`;
   - `VehicleDetector`;
   - `ParkingSlotDetector`;
   - `Tracker`;
   - `OccupancyEstimator`;
   - `TemporalStateManager`;
   - `Visualizer`.

Критерий готовности:

```text
python -m src.main --config configs/default.yaml
```

запускается без падения на тестовом видео, даже если slot detector пока mock.

---

### Этап B — YOLO vehicle detection

1. Подключить Ultralytics YOLO.
2. Загружать `yolo11n.pt` или `yolov8n.pt`.
3. Фильтровать классы:
   - car;
   - truck;
   - bus;
   - motorcycle;
   - person.
4. Рисовать bbox на кадре.
5. Сохранять output video.

Критерий готовности:

```text
на видео видны bbox автомобилей и людей
```

---

### Этап C — tracking

1. Подключить ByteTrack/BoT-SORT через Ultralytics.
2. Получать `track_id`.
3. Считать центр bbox.
4. Считать скорость track по пикселям.

Критерий готовности:

```text
одна и та же машина сохраняет track_id на соседних кадрах
```

---

### Этап D — parking slot detector

1. Подключить pretrained модель из CRPS-D, если доступна.
2. Если формат модели неудобен, временно сделать adapter/mock:
   - загружать заранее подготовленные slot polygons из JSON;
   - но в README явно отметить, что это временная замена.
3. Привести выход к формату `ParkingSlot`.

Критерий готовности:

```text
на кадре рисуются parking slot polygons
```

---

### Этап E — occupancy logic

1. Реализовать `slot_coverage`.
2. Реализовать `point_in_polygon`.
3. Реализовать проверку скорости.
4. Реализовать счетчики `occupied_counter` и `free_counter`.

Критерий готовности:

```text
статус места не прыгает от одного ошибочного кадра
```

---

### Этап F — output

1. Рисовать поверх видео:
   - зеленый slot = free;
   - красный slot = occupied;
   - желтый = potentially_occupied / soon_free.
2. Сохранять JSON:

```json
{
  "frame_id": 125,
  "spots": [
    {
      "slot_id": 1,
      "status": "occupied",
      "confidence": 0.87,
      "assigned_track_id": 12
    }
  ]
}
```

---

## 16. Что считать успешным MVP

MVP успешен, если:

```text
1. Видео читается.
2. Машины детектируются.
3. Машины получают track_id.
4. Parking slots отображаются.
5. Для каждого места вычисляется статус.
6. Статус стабилизируется во времени.
7. Результат сохраняется как видео.
8. Результат сохраняется как JSON.
```

---

## 17. Что НЕ делать в первой версии

Не делать:

```text
- полноценный SLAM;
- 3D-реконструкцию сцены;
- обучение большой модели с нуля;
- сложную разметку собственного большого датасета;
- end-to-end автономную парковку;
- идеальное предсказание освобождения места;
- production-ready real-time server.
```

Главная цель: **рабочий исследовательский прототип**.

---

## 18. Эксперименты для отчета/диплома

Минимальный набор экспериментов:

1. **Без временной логики vs с временной логикой**
   - сравнить стабильность статуса;
   - показать, что счетчики уменьшают ложные переключения.

2. **Обработка каждого кадра vs каждого N-го кадра**
   - сравнить FPS;
   - сравнить качество.

3. **YOLO nano vs small**
   - скорость;
   - качество визуально/по метрикам.

4. **CRPS-D vs PS 2.0**
   - как минимум качественное сравнение применимости;
   - если получится — количественное сравнение.

5. **Разные пороги занятости**
   - `slot_coverage_threshold = 0.15 / 0.20 / 0.30`;
   - `occupied_confirm_frames = 3 / 5 / 10`.

---

## 19. Метрики

Для parking slot detection:

```text
precision
recall
F1
mAP — если модель/датасет позволяет
```

Для occupancy logic:

```text
accuracy по кадрам
число ложных переключений статуса
средняя задержка смены статуса
```

Для real-time:

```text
FPS
latency per frame
GPU memory usage
```

Для трекинга, если есть разметка:

```text
ID switches
track fragmentation
```

---

## 20. Формулировка исследовательской идеи

Основная новизна не в том, чтобы “просто запустить YOLO”.

Более сильная формулировка:

```text
В работе исследуется конвейер определения свободных парковочных мест
с камеры движущегося автомобиля. В отличие от решений для стационарных
камер, система не использует заранее фиксированную разметку парковочных
мест, а выполняет покадровую детекцию parking slots, сопоставляет их с
обнаруженными транспортными средствами и стабилизирует статус занятости
за счет трекинга и временной логики.
```

---

## 21. Короткое описание для README

```text
Проект реализует прототип системы определения свободных парковочных мест
по видео с камеры движущегося автомобиля. Система объединяет детекцию
парковочных мест, детекцию транспортных средств, трекинг объектов и
временную фильтрацию статуса занятости. Основной упор сделан не на обучение
большой модели с нуля, а на построение работающего pipeline, который может
быть запущен на ограниченных вычислительных ресурсах: Kaggle, Colab Free
или локальной GPU начального уровня.
```

---

## 22. Источники и полезные ссылки

### Датасеты

- CRPS-D GitHub: https://github.com/zzh362/CRPS-D
- CRPS-D paper / arXiv: https://arxiv.org/html/2509.13133v1
- PS 2.0 / DeepPS: https://cslinzhang.github.io/deepps/
- Wu & Yeh, Early Detection of Vacant Parking Spaces Using Dashcam Videos: https://ojs.aaai.org/index.php/AAAI/article/view/5024
- PDF Wu & Yeh: https://cdn.aaai.org/ojs/5024/5024-13-8087-1-10-20190709.pdf
- Dragon Lake Parking Dataset: https://sites.google.com/berkeley.edu/dlp-dataset
- DLP API: https://github.com/MPC-Berkeley/dlp-dataset
- ParkingScenes: https://arxiv.org/abs/2604.22835

### Инструменты

- OpenCV VideoCapture tutorial: https://docs.opencv.org/4.x/dd/d43/tutorial_py_video_display.html
- Ultralytics YOLO tracking: https://docs.ultralytics.com/modes/track/
- Ultralytics tracking datasets/trackers: https://docs.ultralytics.com/datasets/track/
- ByteTrack paper: https://arxiv.org/abs/2110.06864

### Вычислительные ресурсы

- Google Colab FAQ: https://research.google.com/colaboratory/faq.html
- Kaggle GPU tips: https://www.kaggle.com/page/GPU-tips-and-tricks
- Ultralytics guide for Kaggle: https://docs.ultralytics.com/integrations/kaggle/

---

## 23. Приоритеты для Codex

Если есть выбор между сложным красивым решением и простым рабочим, выбирать простое рабочее.

Приоритет:

```text
1. Рабочий pipeline.
2. Чистая структура проекта.
3. Понятные интерфейсы классов.
4. Возможность заменить mock на реальную модель.
5. Видео-демонстрация.
6. JSON-вывод.
7. Только потом улучшения и предсказания.
```

Не ломать архитектуру ради одной модели. Все модели должны подключаться через адаптеры.

---

## 24. Главная мысль

Мы делаем не “еще один YOLO-скрипт”, а систему:

```text
moving car video
→ parking slot detection
→ vehicle detection
→ tracking
→ occupancy reasoning
→ stable real-time visualization
```

Это должно быть реализуемо на Colab Free / Kaggle без платных кластеров.
