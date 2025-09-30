# YOLO Database Tools

A comprehensive toolkit for creating custom YOLO datasets and training object detection models with support for both regular and oriented bounding boxes (OBB).

## Overview

This project provides tools to:

- Create annotated datasets for YOLO training with interactive bounding box selection
- Support both oriented bounding boxes (4 points) and regular rectangles
- Train YOLO models using Ultralytics framework
- Handle ellipse and quadrilateral object detection

## Features

- **Interactive Annotation**: Manual bounding box annotation with visual feedback
- **Multiple Bounding Box Types**:
  - Oriented Bounding Box (4 points) - for rotated objects
  - Rectangle (4 points) - for axis-aligned objects
- **Dataset Management**: Automatic train/validation split with proper folder structure
- **YAML Configuration**: Automatic generation of YOLO-compatible dataset configuration
- **Model Training**: Simple interface for training YOLO models with your custom dataset

## Setup

> :warning: Works only on UNIX-Like systems due to the path format used. Only tested on Linux

### 1. Create Virtual Environment

```bash
# Create virtual environment
python -m venv env

# Activate virtual environment
source env/bin/activate  # On Linux/macOS
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

**Key Dependencies:**

- `ultralytics` - YOLO implementation
- `opencv-python` - Image processing
- `matplotlib` - Interactive annotation interface
- `PyYAML` - Configuration file handling
- `torch` & `torchvision` - Deep learning framework

### 3. Prepare Images

Place your images in the `img/` directory. Supported formats:

- `.jpg`, `.jpeg`, `.png`

## Usage

### Step 1: Create Dataset

Run the dataset creation tool:

```bash
python create_ultralytics_dataset.py
```

**Interactive Setup:**

1. **Project Name**: Enter a name for your dataset
2. **Object Name**: Choose or enter the name of objects to detect
   - Suggestions: `ellipse`, `quadrilateral`, `object`
3. **Bounding Box Type**:
   - `1`: Oriented Bounding Box (4 points) - for rotated objects
   - `2`: Rectangle (4 points) - for axis-aligned objects
4. **Training Images**: Specify number of images to annotate for training
5. **Validation Images**: Specify number of images to annotate for validation

**Annotation Process:**

- Images are randomly selected from the `img/` folder
- Interactive annotation interface opens for each image
- Click to define bounding box corners
- For oriented bounding boxes: rotation is enabled
- Options after each annotation:
  - Continue to next image
  - `s` - Delete current annotation and retry
  - `n` or `no` - Stop annotation process

**Output Structure:**

```text
dataset/
└── your_project_name/
    ├── images/
    │   ├── train/
    │   └── val/
    ├── labels/
    │   ├── train/
    │   └── val/
    └── your_project_name.yml
```

### Step 2: Train YOLO Model

Run the training script:

```bash
python train_yolo_model.py
```

**Training Configuration:**

1. **Model Selection**: Choose from available YOLO models:
   - `yolo11s-obb.pt` - For oriented bounding boxes
   - `yolo11s.pt` - For regular bounding boxes
2. **Dataset Selection**: Choose from available datasets in the `dataset/` folder
3. **Training Parameters** (default):
   - Epochs: 250
   - Image size: 640

**Training Output:**

- Trained models saved in `runs/obb/train/weights/` or `runs/detect/train/weights/`
- Best model: `best.pt`
- Last checkpoint: `last.pt`
- Training metrics and logs

### Step 3: Use Trained Model

After training, you can use the model for inference:

```python
from ultralytics import YOLO

# Load your trained model
model = YOLO("runs/obb/train/weights/best.pt")  # or runs/detect/train/weights/best.pt

# Predict on new images
results = model("path/to/your/image.jpg", show=True)

# Access results
for result in results:
    if hasattr(result, 'obb'):  # Oriented bounding boxes
        xyxyxyxy = result.obb.xyxyxyxy  # 4-point polygon format
        confs = result.obb.conf  # confidence scores
        names = [result.names[cls.item()] for cls in result.obb.cls.int()]
    else:  # Regular bounding boxes
        boxes = result.boxes.xyxy  # x1, y1, x2, y2 format
        confs = result.boxes.conf  # confidence scores
        names = [result.names[cls.item()] for cls in result.boxes.cls.int()]
```

## File Structure

```text
yolo_db_tools/
├── README.md                          # This file
├── requirements.txt                   # Python dependencies
├── create_ultralytics_dataset.py      # Dataset creation tool
├── train_yolo_model.py                # Model training script
├── bb_tools.py                        # Bounding box annotation utilities
├── script.ipynb                       # Jupyter notebook for experiments
├── img/                               # Source images for annotation
├── dataset/                           # Generated datasets
│   ├── ellipse_recognition/
│   ├── quadrilateral_recognition/
│   └── your_custom_datasets/
├── runs/                              # Training outputs
│   └── obb/                          # Oriented bounding box results
└── env/                              # Virtual environment
```

## Tips and Best Practices

### Dataset Creation

- **Image Quality**: Use high-quality, diverse images
- **Annotation Accuracy**: Take time to accurately annotate bounding boxes
- **Data Balance**: Ensure good representation of your target objects
- **Train/Val Split**: Typically 80/20 or 70/30 split is recommended

### Training

- **Model Selection**:
  - Use `yolo11s-obb.pt` for objects that may appear rotated
  - Use `yolo11s.pt` for objects that are typically axis-aligned
- **Epochs**: Start with 250 epochs, adjust based on validation metrics
- **Image Size**: 640 is a good default, increase for small objects

### Hardware Requirements

- **GPU**: CUDA-compatible GPU recommended for training
- **Memory**: At least 8GB RAM, 16GB+ recommended for larger datasets
- **Storage**: Ensure adequate space for datasets and training outputs

## Troubleshooting

### Common Issues

1. **"No images found"**: Ensure images are in the `img/` folder with supported formats
2. **CUDA errors**: Check GPU compatibility and CUDA installation
3. **Memory errors**: Reduce batch size or image size in training parameters
4. **Annotation interface not responsive**: Check matplotlib backend configuration

### Getting Help

- Check [Ultralytics documentation](https://docs.ultralytics.com/)
- Verify image formats and dataset structure
- Ensure all dependencies are correctly installed

## Example Workflow

```bash
# 1. Setup environment
python -m venv env
source env/bin/activate
pip install -r requirements.txt

# 2. Add your images to img/ folder
cp /path/to/your/images/* img/

# 3. Create dataset
python create_ultralytics_dataset.py
# Follow interactive prompts

# 4. Train model
python train_yolo_model.py
# Select model and dataset

# 5. Use trained model for inference
# See Step 3 above for code example
```

## License

This project is provided as-is for educational and research purposes.
