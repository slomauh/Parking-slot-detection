# Kaggle Fine-Tune: ParkRecon3D Slot Detector

Нужно загрузить в Kaggle три датасета:

1. `parkrecon3d_bev_crpsd_format.zip`
   - путь локально: `outputs/kaggle_parkrecon3d_bev_dataset/parkrecon3d_bev_crpsd_format.zip`
   - содержит подготовленный ParkRecon3D BEV split.

2. `slot_detector_training_code.zip`
   - путь локально: `outputs/kaggle_slot_detector_training_code/slot_detector_training_code.zip`
   - содержит `train_slot_detector_parkrecon3d.py` и исходники CRPS-D.

3. `crpsd_slot_detector_pretrained_1_2.zip`
   - путь локально: `outputs/kaggle_slot_detector_weights/crpsd_slot_detector_pretrained_1_2.zip`
   - содержит pretrained веса `crpsd_slot_detector_1_2.pth`.

## Kaggle Notebook

Включить GPU: лучше T4 / T4 x2.

Проверить пути:

```bash
!find /kaggle/input -maxdepth 4 -type f | head -80
```

Ожидаемые пути после загрузки датасетов обычно будут похожи на:

```text
/kaggle/input/parkrecon3d-bev-crpsd-format/parkrecon3d_bev_crpsd_format/prepared/train
/kaggle/input/parkrecon3d-bev-crpsd-format/parkrecon3d_bev_crpsd_format/prepared/test
/kaggle/input/slot-detector-training-code/slot_detector_training_code/scripts/train_slot_detector_parkrecon3d.py
/kaggle/input/slot-detector-training-code/slot_detector_training_code/CRPS-D
/kaggle/input/crpsd-slot-detector-pretrained-1-2/crpsd_slot_detector_1_2.pth
```

Если Kaggle назвал датасеты иначе, поменять префиксы в команде.

## Запуск обучения

```bash
!python /kaggle/input/slot-detector-training-code/slot_detector_training_code/scripts/train_slot_detector_parkrecon3d.py \
  --train-dir /kaggle/input/parkrecon3d-bev-crpsd-format/parkrecon3d_bev_crpsd_format/prepared/train \
  --val-dir /kaggle/input/parkrecon3d-bev-crpsd-format/parkrecon3d_bev_crpsd_format/prepared/test \
  --external-repo-path /kaggle/input/slot-detector-training-code/slot_detector_training_code/CRPS-D \
  --pretrained-path /kaggle/input/crpsd-slot-detector-pretrained-1-2/crpsd_slot_detector_1_2.pth \
  --output-path /kaggle/working/parkrecon3d_slot_detector_finetuned.pth \
  --metadata-path /kaggle/working/parkrecon3d_slot_detector_finetuned.json \
  --epochs 20 \
  --batch-size 8 \
  --lr 1e-5 \
  --weight-decay 1e-4 \
  --device cuda \
  --num-workers 2
```

После обучения скачать из `/kaggle/working`:

```text
parkrecon3d_slot_detector_finetuned.pth
parkrecon3d_slot_detector_finetuned.json
```

`.pth` совместим с локальным `ParkingSlotDetector` через `model_path`.
