#!/usr/bin/env python3
"""
Simple Dataset Inspector
Quick functions to check for annotation issues in YOLO datasets
"""

import os
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Polygon
import cv2

def check_annotation_issues(dataset_path, split='train'):
    """
    Quick check for annotation issues in a dataset
    Returns summary of problems found
    """
    dataset_path = Path(dataset_path)
    labels_dir = dataset_path / 'labels' / split
    
    if not labels_dir.exists():
        print(f"Labels directory not found: {labels_dir}")
        return
    
    issues = {
        'negative_dimensions': [],
        'out_of_bounds': [],
        'invalid_format': [],
        'empty_files': []
    }
    
    label_files = list(labels_dir.glob('*.txt'))
    print(f"Checking {len(label_files)} label files...")
    
    for label_file in label_files:
        if label_file.stat().st_size == 0:
            issues['empty_files'].append(label_file.name)
            continue
            
        with open(label_file, 'r') as f:
            for line_num, line in enumerate(f, 1):
                if not line.strip():
                    continue
                    
                parts = line.strip().split()
                if len(parts) < 5:
                    issues['invalid_format'].append(f"{label_file.name}:{line_num}")
                    continue
                
                try:
                    coords = [float(x) for x in parts[1:]]
                except ValueError:
                    issues['invalid_format'].append(f"{label_file.name}:{line_num}")
                    continue
                
                # Check different formats
                if len(coords) == 8:  # OBB format
                    # Check if coordinates are within bounds
                    if any(c < 0 or c > 1 for c in coords):
                        issues['out_of_bounds'].append(f"{label_file.name}:{line_num}")
                    
                    # Check for negative dimensions
                    points = [(coords[i], coords[i + 1]) for i in range(0, 8, 2)]
                    x_coords = [p[0] for p in points]
                    y_coords = [p[1] for p in points]
                    width = max(x_coords) - min(x_coords)
                    height = max(y_coords) - min(y_coords)
                    
                    if width <= 0 or height <= 0:
                        issues['negative_dimensions'].append(f"{label_file.name}:{line_num} (w={width:.3f}, h={height:.3f})")
                
                elif len(coords) >= 4:  # Regular bbox format
                    if len(coords) == 4:  # x_center, y_center, width, height
                        x_center, y_center, width, height = coords[:4]
                    elif len(coords) == 5:  # x_center, y_center, width, height, angle
                        x_center, y_center, width, height = coords[:4]
                    
                    # Check bounds
                    if x_center < 0 or x_center > 1 or y_center < 0 or y_center > 1:
                        issues['out_of_bounds'].append(f"{label_file.name}:{line_num}")
                    
                    # Check negative dimensions
                    if width <= 0 or height <= 0:
                        issues['negative_dimensions'].append(f"{label_file.name}:{line_num} (w={width:.3f}, h={height:.3f})")
    
    # Print summary
    print("\\nAnnotation Issues Summary:")
    print("=" * 40)
    
    total_issues = sum(len(issue_list) for issue_list in issues.values())
    
    if total_issues == 0:
        print("✅ No issues found!")
    else:
        print(f"❌ Found {total_issues} issues:")
        
        if issues['negative_dimensions']:
            print(f"\\n🔴 Negative/zero dimensions ({len(issues['negative_dimensions'])}):")
            for issue in issues['negative_dimensions'][:10]:  # Show first 10
                print(f"  - {issue}")
            if len(issues['negative_dimensions']) > 10:
                print(f"  ... and {len(issues['negative_dimensions']) - 10} more")
        
        if issues['out_of_bounds']:
            print(f"\\n🟡 Out of bounds coordinates ({len(issues['out_of_bounds'])}):")
            for issue in issues['out_of_bounds'][:5]:
                print(f"  - {issue}")
            if len(issues['out_of_bounds']) > 5:
                print(f"  ... and {len(issues['out_of_bounds']) - 5} more")
        
        if issues['invalid_format']:
            print(f"\\n🟠 Invalid format ({len(issues['invalid_format'])}):")
            for issue in issues['invalid_format'][:5]:
                print(f"  - {issue}")
            if len(issues['invalid_format']) > 5:
                print(f"  ... and {len(issues['invalid_format']) - 5} more")
        
        if issues['empty_files']:
            print(f"\\n⚪ Empty files ({len(issues['empty_files'])}):")
            for issue in issues['empty_files'][:5]:
                print(f"  - {issue}")
            if len(issues['empty_files']) > 5:
                print(f"  ... and {len(issues['empty_files']) - 5} more")
    
    return issues

def quick_view_issues(dataset_path, split='train', max_show=5):
    """
    Quickly view a few images that have annotation issues
    """
    from dataset_viewer import DatasetViewer
    
    print(f"Quick viewing of annotation issues in {dataset_path}/{split}...")
    
    # First, check for issues
    issues = check_annotation_issues(dataset_path, split)
    
    if not any(issues.values()):
        print("No issues found to display!")
        return
    
    # Get files with negative dimension issues
    problem_files = []
    for issue in issues['negative_dimensions']:
        file_name = issue.split(':')[0]
        if file_name not in problem_files:
            problem_files.append(file_name)
    
    if not problem_files:
        print("No files with negative dimensions found!")
        return
    
    print(f"\\nShowing up to {min(max_show, len(problem_files))} files with issues...")
    
    viewer = DatasetViewer(dataset_path)
    
    dataset_path = Path(dataset_path)
    images_dir = dataset_path / 'images' / split
    labels_dir = dataset_path / 'labels' / split
    
    shown = 0
    for file_name in problem_files[:max_show]:
        # Find corresponding image file
        image_name = file_name.replace('.txt', '')
        
        # Try different extensions
        image_path = None
        for ext in ['.jpg', '.jpeg', '.png', '.bmp']:
            potential_path = images_dir / (image_name + ext)
            if potential_path.exists():
                image_path = potential_path
                break
        
        if image_path:
            label_path = labels_dir / file_name
            print(f"\\nShowing issue in: {image_path.name}")
            viewer.view_image(image_path, label_path)
            shown += 1
            
            if shown < min(max_show, len(problem_files)):
                input("Press Enter to see next image with issues...")
        else:
            print(f"Could not find image for {file_name}")
    
    print(f"\\nShowed {shown} images with annotation issues.")

def view_all_images(dataset_path, split='train', max_images=None):
    """
    View images from the dataset split with all bounding boxes overlaid.
    This leverages the existing DatasetViewer.
    """
    from dataset_viewer import DatasetViewer
    print(f"Viewing all images in {dataset_path}/{split}...")
    viewer = DatasetViewer(dataset_path)
    viewer.view_dataset(split=split, max_images=max_images, show_issues_only=False)

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python dataset_inspector.py <dataset_path> [split]")
        print("Example: python dataset_inspector.py dataset/ellipse_recognition train")
        sys.exit(1)
    
    dataset_path = sys.argv[1]
    split = sys.argv[2] if len(sys.argv) > 2 else 'train'
    
    # Quick check for issues
    check_annotation_issues(dataset_path, split)
    
    # Ask if user wants to see visual examples
    response = input("\\nDo you want to see visual examples of the issues? (y/n): ")
    if response.lower().startswith('y'):
        quick_view_issues(dataset_path, split)

    # Optional: view all images with bounding boxes
    response_all = input("\\nDo you want to view all images with bounding boxes? (y/n): ")
    if response_all.lower().startswith('y'):
        max_images_input = input("Maximum number of images to show (press Enter for all): ").strip()
        max_images = int(max_images_input) if max_images_input.isdigit() else None
        view_all_images(dataset_path, split, max_images)