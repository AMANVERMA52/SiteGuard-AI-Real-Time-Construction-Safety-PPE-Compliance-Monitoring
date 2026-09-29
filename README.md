# SiteGuard AI: Real-Time Construction Safety & PPE Compliance Monitoring

A computer vision system for detecting personal protective equipment (PPE) and construction-site objects using a custom-trained YOLOv8 model. The repository includes model training infrastructure, dataset organization, inference utilities, and a Flask-based real-time monitoring application.

The deployment pipeline combines object detection with configurable PPE-compliance logic, event persistence, snapshot capture, and a browser-based monitoring interface.

## Overview

The system performs object detection on construction-site imagery and video streams and identifies the following classes:

* `Hardhat`
* `Mask`
* `NO-Hardhat`
* `NO-Mask`
* `NO-Safety Vest`
* `Person`
* `Safety Cone`
* `Safety Vest`
* `machinery`
* `vehicle`

The detection model is based on **YOLOv8s** and was fine-tuned using the Construction Site Safety Image Dataset. The repository also provides a Flask application for real-time camera inference and PPE-compliance monitoring.

The system supports three primary workflows:

1. Static-image inference
2. Real-time camera inference
3. Browser-based PPE compliance monitoring

## System Architecture

The application is organized into three major layers:

### 1. Model Training

The training pipeline uses Ultralytics YOLOv8 and a YOLO-format construction-site dataset.

```text
Dataset
   │
   ├── Training Images
   ├── Validation Images
   └── Test Images
          │
          ▼
     YOLOv8s Transfer Learning
          │
          ▼
     Trained Model Weights
          │
          ▼
   Inference / Deployment
```

### 2. Inference

The inference layer loads the trained YOLO model and processes images or camera frames.

For every frame, the model produces:

* Bounding boxes
* Class labels
* Confidence scores

The deployment application converts these raw detections into a structured detection representation before passing them to the PPE-compliance layer.

### 3. Compliance Monitoring

The deployment layer evaluates detected PPE against configurable requirements.

```text
Camera Frame
     │
     ▼
YOLOv8 Inference
     │
     ▼
Object Detections
     │
     ▼
InstanceDetector
     │
     ▼
ComplianceChecker
     │
     ├── Compliant
     │
     └── Non-Compliant
             │
             ├── Alert
             ├── Snapshot
             └── Database Record
```

The Flask application exposes the processed results through a browser dashboard and Socket.IO events.

---

## Dataset

The project uses the Construction Site Safety Image Dataset sourced from Roboflow.

Dataset distribution:

| Split      | Images |
| ---------- | -----: |
| Training   |  2,605 |
| Validation |    114 |
| Test       |     82 |
| Total      |  2,801 |

Annotations use the YOLO format:

```text
class_id center_x center_y width height
```

Coordinates are normalized relative to image dimensions.

The dataset contains 10 object classes covering PPE, PPE violations, workers, and construction-site equipment.

## Detection Classes

| Category       | Classes                                         |
| -------------- | ----------------------------------------------- |
| PPE            | `Hardhat`, `Mask`, `Safety Vest`, `Safety Cone` |
| PPE Violations | `NO-Hardhat`, `NO-Mask`, `NO-Safety Vest`       |
| People         | `Person`                                        |
| Equipment      | `machinery`, `vehicle`                          |

The model therefore provides both positive PPE detections and explicit negative PPE classes.

---

# Model

## Architecture

The primary detector is **YOLOv8s**, the Small configuration of the YOLOv8 object-detection family.

The model was initialized from pretrained weights and fine-tuned for the construction-site PPE dataset.

### Training configuration

| Parameter             | Value              |
| --------------------- | ------------------ |
| Architecture          | YOLOv8s            |
| Epochs                | 200                |
| Batch size            | 16                 |
| Base learning rate    | `1e-3`             |
| Optimizer             | Auto               |
| Training GPU          | NVIDIA RTX 5060 Ti |
| GPU memory            | 16 GB              |
| Approx. training time | 1.92 hours         |

The repository reports the following final training metrics:

| Metric    |  Value |
| --------- | -----: |
| Precision | 0.9500 |
| Recall    | 0.7975 |
| mAP@50    | 0.8767 |
| mAP@50–95 | 0.6152 |

These metrics are the values reported by the repository's training run and should be interpreted as dataset-specific evaluation results rather than a guarantee of performance on unseen construction environments.

---

# Training Pipeline

The training workflow is implemented in:

```text
Model-Training/yolov8-finetuning-for-ppe-detection.ipynb
```

The notebook covers:

1. Dataset loading
2. Dataset validation
3. YOLO configuration
4. YOLOv8s initialization
5. Transfer learning
6. Model training
7. Validation
8. Metric collection
9. Model checkpointing
10. Training-result analysis

The dataset configuration is stored in:

```text
Model-Training/Outputs/data.yaml
```

The configuration defines the dataset paths and the ten detection classes.

### Training outputs

Training artifacts are stored under:

```text
Model-Training/Outputs/runs/detect/
```

The repository includes epoch-level training information such as:

* Box loss
* Classification loss
* Distribution focal loss
* Precision
* Recall
* mAP@50
* mAP@50–95
* Validation losses
* Learning-rate values

---

# Inference

The project provides standalone inference scripts under:

```text
Model-Testing/
```

## Hardware check

Run:

```bash
python Model-Testing/checker.py
```

## Webcam inference

Run:

```bash
python Model-Testing/test_cam.py
```

The webcam pipeline:

1. Captures a frame using OpenCV.
2. Passes the frame to YOLO.
3. Runs object detection.
4. Extracts class labels and confidence scores.
5. Renders bounding boxes and labels.
6. Displays the annotated frame.

## Single-image inference

Run:

```bash
python Model-Testing/test_single.py
```

This provides a lightweight way to test the model against individual images without starting the web application.

---

# Web Application

The deployment implementation is located in:

```text
Model-Deployment/
```

The application uses:

* Flask
* Flask-SocketIO
* OpenCV
* Ultralytics YOLO
* SQLite
* Python threading

The dependency list is defined in `requirements.txt`.

## Application Components

### `app.py`

`app.py` is the main application entry point.

Responsibilities include:

* Flask server initialization
* YOLO model loading
* Camera initialization
* Video-frame processing
* Detection processing
* Compliance evaluation
* Snapshot generation
* Database logging
* REST API endpoints
* Socket.IO event emission

The application processes each camera frame through the following sequence:

```text
OpenCV Camera
      │
      ▼
Frame Capture
      │
      ▼
YOLO Model
      │
      ▼
Raw Detections
      │
      ▼
InstanceDetector
      │
      ▼
ComplianceChecker
      │
      ├───────────────┐
      ▼               ▼
Compliant       Non-Compliant
                      │
              ┌───────┼───────┐
              ▼       ▼       ▼
            Alert  Snapshot  SQLite
```

The current implementation exposes endpoints for streaming, camera control, settings, statistics, instance history, and snapshot retrieval.

---

# PPE Compliance Engine

The compliance logic is implemented in:

```text
Model-Deployment/detection_logic.py
```

The main components are:

* `InstanceDetector`
* `ComplianceChecker`
* `SnapshotManager`

## InstanceDetector

`InstanceDetector` maintains the current detection state and evaluates PPE requirements.

It supports configurable requirements for:

```json
{
  "helmet": true,
  "vest": true,
  "mask": false
}
```

It also maintains:

* Instance identifiers
* Detection state
* Non-compliance timing
* Instance timeout
* Snapshot state
* Detection mode

Instance identifiers use a date/serial-number format.

Example:

```text
09_28_2026_12
```

The serial counter is persisted in:

```text
instance_counter.json
```

and is reset for a new calendar day.

---

# Detection Modes

The application supports two compliance modes.

## Single-Person Mode

In single-person mode, the system checks whether the configured PPE classes are present among the detections.

For example, if the configuration requires:

```text
Helmet = required
Vest   = required
Mask   = optional
```

the compliance engine checks for the corresponding detected PPE classes.

The implementation currently treats PPE presence at the frame/detection-set level rather than performing sophisticated geometric association between each PPE item and a specific worker.

## Multi-Person Mode

Multi-person mode uses explicit negative PPE detections such as:

```text
NO-Hardhat
NO-Safety Vest
NO-Mask
```

to identify non-compliance when multiple people are present.

The implementation checks whether required PPE violation classes are present in the current detection set.

---

# Non-Compliance Detection

To reduce alerts caused by transient detection errors, the system does not immediately classify a single non-compliant frame as a violation.

A configurable delay is applied before the system enters a confirmed non-compliance state.

The default delay is:

```text
3 seconds
```

If the detected violation persists beyond the configured delay, the instance is marked non-compliant.

This mechanism is implemented in `InstanceDetector`.

---

# Alerting

When confirmed non-compliance is detected, the application:

1. Marks the current instance as non-compliant.
2. Creates an instance identifier.
3. Logs an alert in SQLite.
4. Emits a Socket.IO `alert` event.
5. Generates detection-update events for the dashboard.
6. Captures a snapshot when configured by the detection state.

The application also applies an alert cooldown to avoid repeatedly generating alerts for the same continuous violation.

---

# Snapshot Management

Non-compliant detections can be persisted as JPEG snapshots.

Snapshots are stored under:

```text
Model-Deployment/snapshots/
```

The `SnapshotManager` creates the directory automatically when necessary and writes captured frames as JPEG files.

Snapshots are associated with detection instances and can subsequently be retrieved through the web application.

---

# Persistence Layer

The application uses SQLite for local persistence.

The database stores information associated with:

* Detection instances
* PPE compliance events
* Alerts
* Snapshots
* Historical detection records

The database implementation is contained in:

```text
Model-Deployment/database.py
```

The default database file is:

```text
Model-Deployment/detections.db
```

This allows detection information to remain available after the application process is restarted.

---

# Configuration

Runtime configuration is stored in:

```text
Model-Deployment/settings.json
```

Important parameters include:

### Required PPE

```json
"required_ppe": {
    "helmet": true,
    "vest": true,
    "mask": false
}
```

### Non-compliance delay

```json
"non_compliance_delay": 3
```

The value represents the number of seconds a violation must persist before being classified as confirmed non-compliance.

### Instance reset timeout

```json
"instance_reset_timeout": 5
```

This controls when an inactive detection instance is reset.

### Detection mode

```json
"detection_mode": "single"
```

Supported values are:

```text
single
multi
```

The application also provides a browser-based settings interface for changing supported configuration values.

---

# Web Interface

The Flask application contains three primary interfaces.

## Dashboard

```text
/
```

The dashboard provides:

* Live camera stream
* Detection overlays
* PPE status
* Detection statistics
* Compliance state
* Alert notifications

The browser receives real-time updates through Socket.IO.


## Requirements

Recommended environment:

* Python 3.9+
* pip
* OpenCV-compatible camera for live inference
* CPU or CUDA-compatible GPU
* Windows, Linux, or macOS

For GPU acceleration, an appropriate NVIDIA driver and CUDA-compatible PyTorch installation are required.

### Windows

```powershell
python -m venv .venv
.venv\Scripts\activate
```

### Linux/macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
```

## Install dependencies

```bash
pip install -r requirements.txt
```

The current requirements file contains:

```text
Flask
Flask-SocketIO
opencv-python
ultralytics
Pillow
python-socketio
eventlet
```

# Model Path Configuration

The deployment application currently defines its model path in `Model-Deployment/app.py`.

```text
yolov8s_ppe_css_200_epochs
```

Therefore, deployments using the 200-epoch model should update the path to the corresponding `best.pt` file before running the web application.

If the configured model path does not exist, the current application falls back to:

```python
YOLO('yolov8n.pt')
```

This fallback should not be interpreted as the custom PPE model.

---


# Performance Considerations

Inference performance depends on:

* Model size
* Input resolution
* CPU/GPU hardware
* PyTorch configuration
* Camera resolution
* Number of objects in each frame
* Post-processing overhead

YOLOv8s provides a compromise between model capacity and inference cost. For lower-resource deployments, a smaller YOLO model can be evaluated, although changing the model requires verifying compatibility with the project's custom PPE classes.

For real-time deployment, inference latency and effective frames per second should be measured on the target hardware rather than inferred from training time.

---

# Limitations

The current implementation has several technical limitations that should be considered before production deployment.

## Per-worker PPE association

The compliance logic does not implement a dedicated multi-object tracking algorithm such as DeepSORT or ByteTrack.

In particular, the single-person mode evaluates PPE detections at the current detection-set level rather than performing robust spatial association between each PPE item and an individual worker.

Consequently, multi-worker scenes may require additional association and tracking logic for reliable per-worker compliance decisions.

## Detection dependence

Compliance decisions depend directly on YOLO detection results.

Missed detections can therefore produce false compliance or non-compliance states depending on the detected class configuration.

## Camera conditions

Performance can degrade under:

* Poor illumination
* Occlusion
* Motion blur
* Significant camera-angle changes
* Small or distant workers
* PPE visually different from the training distribution

## Dataset generalization

The reported metrics are based on the project's construction-site dataset. Real-world performance should be validated against representative deployment environments before operational use.

## Local deployment architecture

The current application is designed as a local Flask/Socket.IO application with SQLite persistence. It is not, by itself, a horizontally scalable multi-camera production architecture.

---

# Training vs. Inference

Retraining is not required to run the inference or dashboard components if the trained model weights are available.

The intended workflow for using the existing model is:

```text
Install dependencies
       │
       ▼
Configure model path
       │
       ▼
Run inference
       │
       ├── Static image
       ├── Webcam
       └── Flask dashboard

# Reproducibility

For reproducible experiments, record:

* Python version
* PyTorch version
* Ultralytics version
* CUDA version, when applicable
* GPU model
* Dataset revision
* Model checkpoint
* Image resolution
* Batch size
* Learning rate
* Number of epochs
* Confidence threshold
* IoU threshold

The repository contains training logs and model artifacts that can be used for retrospective analysis of the reported training run.

---

# Technology Stack

| Component               | Technology                             |
| ----------------------- | -------------------------------------- |
| Object Detection        | YOLOv8                                 |
| Deep Learning           | PyTorch / Ultralytics                  |
| Image Processing        | OpenCV                                 |
| Web Framework           | Flask                                  |
| Real-Time Communication | Flask-SocketIO                         |
| Database                | SQLite                                 |
| Image Handling          | Pillow                                 |
| Numerical Processing    | NumPy                                  |
| Dataset                 | Construction Site Safety Image Dataset |
| Annotation Format       | YOLO                                   |

---

# Development Roadmap

Potential engineering improvements include:

* Per-worker PPE association
* Dedicated multi-object tracking
* Improved temporal filtering
* Multi-camera processing
* GPU/CPU inference optimization
* Model quantization
* ONNX export
* Edge-device deployment
* Centralized event storage
* Authentication and access control
* Production-grade API design
* Containerized deployment
* Automated testing and CI/CD

These are development opportunities rather than components currently guaranteed by the implementation.

---

# Summary

This repository implements an end-to-end construction-site PPE detection pipeline using a custom YOLOv8s object detector.

The system covers:

* Dataset preparation
* YOLOv8 transfer learning
* Model evaluation
* Static-image inference
* Real-time camera inference
* PPE-compliance evaluation
* Configurable violation timing
* Instance identification
* Non-compliance alerts
* Snapshot capture
* SQLite persistence
* Flask web deployment
* Socket.IO real-time updates
* Historical detection review

The architecture is intended to provide a complete path from object detection to local real-time PPE monitoring while keeping the model, inference logic, compliance engine, and web application separated into distinct components.
