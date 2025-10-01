# Dataset Viewer and Inspector Guide

## Overview

I've created two tools to help you review and debug your YOLO dataset annotations:

1. **`dataset_inspector.py`** - Quick command-line tool to find annotation issues
2. **`dataset_viewer.py`** - Full visual dataset viewer with annotation display

## Quick Start

### 1. Check for Annotation Issues (Fast)

```bash
# Check for problems in your dataset
python dataset_inspector.py dataset/your_dataset_name train

# Examples:
python dataset_inspector.py dataset/ellipse_recognition train
python dataset_inspector.py dataset/test_12 train
python dataset_inspector.py dataset/test_12 val
```

This will quickly scan all label files and report:
- ❌ **Negative dimensions** (main issue you're experiencing)
- 🟡 **Out of bounds coordinates** (coordinates > 1.0)
- 🟠 **Invalid format** (wrong number of coordinates)
- ⚪ **Empty files**

### 2. Visual Review (Detailed)

```bash
# View dataset with bounding box visualization
python dataset_viewer.py dataset/your_dataset_name

# Options:
python dataset_viewer.py dataset/test_12 --split train
python dataset_viewer.py dataset/test_12 --split val --issues-only
python dataset_viewer.py dataset/test_12 --max-images 10
```

## Features

### Dataset Inspector Features
- **Fast scanning** of all annotation files
- **Issue categorization** with counts and examples  
- **Quick visual preview** of problematic annotations
- **Summary statistics** including issue rates

### Dataset Viewer Features  
- **Full visualization** of images with bounding boxes
- **Multiple format support**: 
  - Oriented Bounding Boxes (8 coordinates)
  - Regular bounding boxes (4 coordinates) 
  - Rotated bounding boxes (5 coordinates)
- **Issue highlighting**: Red boxes for negative dimensions
- **Interactive browsing**: Navigate through images
- **Filter options**: Show only problematic images

## Understanding the Issues

### Negative Dimensions Issue
When you see: `w=-0.292, h=-0.483`

This happens when the bounding box coordinates are in the wrong order. For oriented bounding boxes, the 4 points should be in a consistent order (clockwise or counter-clockwise).

**Example of good OBB coordinates:**
```
0 0.1 0.1 0.9 0.1 0.9 0.9 0.1 0.9  # Rectangle: top-left, top-right, bottom-right, bottom-left
```

**Example of bad OBB coordinates:**
```
0 0.1 0.1 0.9 0.9 0.9 0.1 0.1 0.9  # Points in wrong order -> negative dimensions
```

## Example Usage

### Quick Issue Check
```bash
cd /home/paul/dev/projects/yolo_db_tools
source env/bin/activate
python dataset_inspector.py dataset/test_12 train
```

### Visual Review of Problems
```bash
cd /home/paul/dev/projects/yolo_db_tools  
source env/bin/activate
python dataset_viewer.py dataset/test_12 --issues-only
```

### Browse All Images
```bash
cd /home/paul/dev/projects/yolo_db_tools
source env/bin/activate  
python dataset_viewer.py dataset/ellipse_recognition --split train
```

## Fixing the Issues

The negative dimensions are likely caused by the annotation process creating points in the wrong order. You may need to:

1. **Review the `bb_picker` function** in `bb_tools.py` to ensure points are always in the correct order
2. **Add validation** to the annotation process to check for negative dimensions before saving
3. **Re-annotate** the problematic images, or 
4. **Write a script** to automatically fix the point ordering

## Navigation Controls

In the visual viewer:
- **Enter**: Next image
- **'q'**: Quit
- **'s'**: Switch to issues-only mode

## Output Interpretation

### Inspector Output
```
❌ Found 2 issues:
🔴 Negative/zero dimensions (2):
  - filename.txt:1 (w=0.326, h=-0.292)
```

This means:
- File: `filename.txt` 
- Line: `1` (first annotation in the file)
- Width: `0.326` (positive - good)
- Height: `-0.292` (negative - problem!)

### Visual Output
- **Green boxes**: Normal annotations
- **Red boxes**: Annotations with issues
- **Text labels**: Show class name and dimensions if problematic

The tools will help you quickly identify which images and annotations have problems so you can fix them efficiently!