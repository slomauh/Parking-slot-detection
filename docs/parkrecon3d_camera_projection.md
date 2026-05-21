# ParkRecon3D BEV to Camera Projection Experiment

ParkRecon3D camera folders do not contain slot labels directly, but the dataset provides calibration files separately:

```text
param.yaml
stitch.json
coordinate.png
```

Local copies used for the experiment:

```text
external/parkrecon3d_calibration/param.yaml
external/parkrecon3d_calibration/stitch.json
external/parkrecon3d_calibration/coordinate.png
```

Source:

```text
https://huggingface.co/datasets/pipa83165/ParkRecon3D/tree/main
```

## What Was Tested

Script:

```text
scripts/project_parkrecon3d_bev_to_cameras.py
```

Example command:

```bash
python scripts/project_parkrecon3d_bev_to_cameras.py \
  --dataset-root /home/slomauh/Documents/data1 \
  --stitch-json external/parkrecon3d_calibration/stitch.json \
  --output-dir outputs/parkrecon3d_projection_selected \
  --cameras 0 1 2 \
  --timestamps 1585161072082 1662732066896 \
  --mode camera_to_vehicle \
  --swap-xy
```

Debug outputs:

```text
outputs/parkrecon3d_projection_selected/
outputs/parkrecon3d_projection_selected/1662732066896_contact_sheet.jpg
```

## Current Finding

Projection is possible in principle:

```text
BEV pixel -> vehicle ground-plane coordinate -> fisheye camera pixel
```

The best quick-tested convention was:

```text
mode: camera_to_vehicle
axis option: --swap-xy
```

On `Camera1` this gives a visually plausible overlay for many slot lines. On `Camera0` and `Camera2`, some projected lines land on the hood / non-useful part of the image or correspond to slots better seen by another camera.

## Why This Is Not Yet a Training Dataset

Before generating camera labels for detector fine-tuning, the projection needs filtering:

- remove points outside the image;
- remove lines behind the camera or crossing invalid fisheye regions;
- remove slots occluded by the ego vehicle hood/body;
- assign each BEV slot only to the camera where it is actually visible;
- manually inspect a sample before using labels for training.

Without this filtering, projected labels can be noisy and may hurt training.
