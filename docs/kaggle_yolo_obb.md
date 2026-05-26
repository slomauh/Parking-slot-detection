# Kaggle: YOLO-OBB Slot Detector

Этот эксперимент заменяет CRPS-D marking-point pairing на прямую детекцию парковочного места как oriented bounding box.

## Что загрузить в Kaggle

Загрузить как Kaggle Dataset:

```text
outputs/kaggle_parkrecon3d_yolo_obb/parkrecon3d_yolo_obb.zip
```

Внутри будет:

```text
parkrecon3d_yolo_obb/
  images/train
  images/val
  labels/train
  labels/val
  parkrecon3d_obb.yaml
```

## Проверить пути

В Kaggle Notebook:

```bash
!find /kaggle/input -maxdepth 4 -type f | head -80
```

Ожидаемый путь обычно будет похож на:

```text
/kaggle/input/parkrecon3d-yolo-obb/parkrecon3d_yolo_obb/parkrecon3d_obb.yaml
```

Если Kaggle назвал dataset иначе, поменять префикс в команде обучения.

## Smoke train

Сначала создать Kaggle-local yaml с абсолютным путем к датасету. Это нужно, потому что `path: .` внутри загруженного yaml может резолвиться в `/kaggle/working`, а не в `/kaggle/input`.

```python
from pathlib import Path
import yaml

dataset_root = Path("/kaggle/input/parkrecon3d-yolo-obb/parkrecon3d_yolo_obb")
data = "/kaggle/working/parkrecon3d_obb_kaggle.yaml"

with open(data, "w") as f:
    yaml.safe_dump(
        {
            "path": str(dataset_root),
            "train": "images/train",
            "val": "images/val",
            "names": {0: "parking_slot"},
        },
        f,
        sort_keys=False,
    )

print(Path(data).read_text())
print((dataset_root / "images/train").exists(), (dataset_root / "images/val").exists())
```

Потом короткий запуск, чтобы проверить, что YOLO читает OBB labels:

```python
from ultralytics import YOLO

try:
    model = YOLO("yolo26n-obb.pt")
except Exception:
    model = YOLO("yolo11n-obb.pt")

model.train(
    data=data,
    epochs=15,
    imgsz=1024,
    batch=8,
    device=0,
    project="/kaggle/working/yolo_obb_smoke",
    name="parkrecon3d",
)
```

## Full train

```python
from pathlib import Path
import yaml
from ultralytics import YOLO

dataset_root = Path("/kaggle/input/parkrecon3d-yolo-obb/parkrecon3d_yolo_obb")
data = "/kaggle/working/parkrecon3d_obb_kaggle.yaml"

with open(data, "w") as f:
    yaml.safe_dump(
        {
            "path": str(dataset_root),
            "train": "images/train",
            "val": "images/val",
            "names": {0: "parking_slot"},
        },
        f,
        sort_keys=False,
    )

try:
    model = YOLO("yolo26n-obb.pt")
except Exception:
    model = YOLO("yolo11n-obb.pt")

model.train(
    data=data,
    epochs=80,
    imgsz=1024,
    batch=8,
    device=0,
    project="/kaggle/working/yolo_obb",
    name="parkrecon3d",
    patience=20,
)
```

После обучения скачать:

```text
/kaggle/working/yolo_obb/parkrecon3d/weights/best.pt
```

Локально положить, например:

```text
models/slot_detector/parkrecon3d_yolo_obb.pt
```

## Локальная проверка

```bash
python scripts/evaluate_parkrecon3d_bev.py \
  --image-dir outputs/parkrecon3d_bev_crpsd_format/raw/test/img \
  --label-dir outputs/parkrecon3d_bev_crpsd_format/raw/test/slot_label \
  --slot-backend yolo_obb \
  --slot-model-path models/slot_detector/parkrecon3d_yolo_obb.pt \
  --slot-conf 0.25 \
  --detector-input-size 1024 \
  --device cuda \
  --skip-occupancy \
  --output-dir outputs/parkrecon3d_yolo_obb_eval
```

Текущий CRPS-D baseline для сравнения:

```text
recall:    77.48%
precision: 77.54%
F1:        77.51%
```

Если YOLO-OBB дает F1 выше baseline или визуально сильно меньше поперечных слотов/дублей при близком F1, его стоит делать основным backend для slot detector.
