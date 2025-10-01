#!/usr/bin/env python3
"""
Dataset Viewer for Ultralytics YOLO Datasets
Displays images with their bounding boxes to verify annotations
Supports both Oriented Bounding Boxes (OBB) and regular bounding boxes
"""

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Polygon
import argparse
import yaml
from pathlib import Path
import glob

class DatasetViewer:
    def __init__(self, dataset_path):
        self.dataset_path = Path(dataset_path)
        self.yaml_file = None
        self.dataset_info = None
        self.load_dataset_config()
    
    def load_dataset_config(self):
        """Load dataset configuration from YAML file"""
        # Look for YAML files in the dataset directory
        yaml_files = list(self.dataset_path.glob("*.yaml")) + list(self.dataset_path.glob("*.yml"))
        
        if yaml_files:
            self.yaml_file = yaml_files[0]
            with open(self.yaml_file, 'r') as f:
                self.dataset_info = yaml.safe_load(f)
            print(f"Loaded dataset config: {self.yaml_file}")
            print(f"Classes: {self.dataset_info.get('names', ['Unknown'])}")
        else:
            print("No YAML config found, using default settings")
            self.dataset_info = {'names': ['object']}
    
    def parse_label_line(self, line):
        """Parse a single line from a label file"""
        parts = line.strip().split()
        if len(parts) == 0:
            return None
        
        class_id = int(parts[0])
        coords = [float(x) for x in parts[1:]]
        
        if len(coords) == 8:
            # Oriented Bounding Box format: class x1 y1 x2 y2 x3 y3 x4 y4
            return {
                'class_id': class_id,
                'format': 'obb',
                'coords': coords
            }
        elif len(coords) == 4:
            # Regular bounding box format: class x_center y_center width height
            return {
                'class_id': class_id,
                'format': 'bbox',
                'coords': coords
            }
        elif len(coords) == 5:
            # Rotated bounding box format: class x_center y_center width height angle
            return {
                'class_id': class_id,
                'format': 'rotated_bbox',
                'coords': coords
            }
        else:
            print(f"Unknown annotation format with {len(coords)} coordinates: {line}")
            return None
    
    def load_annotations(self, label_file):
        """Load annotations from a label file"""
        annotations = []
        if not label_file.exists():
            return annotations
        
        with open(label_file, 'r') as f:
            for line_num, line in enumerate(f, 1):
                if line.strip():
                    annotation = self.parse_label_line(line)
                    if annotation:
                        annotations.append(annotation)
                    else:
                        print(f"Failed to parse line {line_num} in {label_file}: {line.strip()}")
        
        return annotations
    
    def denormalize_coords(self, coords, img_width, img_height):
        """Convert normalized coordinates to pixel coordinates"""
        denorm_coords = []
        for i in range(0, len(coords), 2):
            x = coords[i] * img_width
            y = coords[i + 1] * img_height
            denorm_coords.extend([x, y])
        return denorm_coords
    
    def draw_obb(self, ax, coords, img_width, img_height, class_id, class_names):
        """Draw oriented bounding box"""
        # Denormalize coordinates
        denorm_coords = self.denormalize_coords(coords, img_width, img_height)
        
        # Create polygon points
        points = [(denorm_coords[i], denorm_coords[i + 1]) for i in range(0, 8, 2)]
        
        # Check for negative dimensions (indicates annotation issues)
        x_coords = [p[0] for p in points]
        y_coords = [p[1] for p in points]
        width = max(x_coords) - min(x_coords)
        height = max(y_coords) - min(y_coords)
        
        color = 'red' if width < 0 or height < 0 else 'lime'
        
        # Draw polygon
        polygon = Polygon(points, linewidth=2, edgecolor=color, facecolor='none')
        ax.add_patch(polygon)
        
        # Add class label
        center_x = sum(x_coords) / 4
        center_y = sum(y_coords) / 4
        class_name = class_names[class_id] if class_id < len(class_names) else f"class_{class_id}"
        
        # Add dimension info if negative
        label = class_name
        if width < 0 or height < 0:
            label += f" (W:{width:.1f}, H:{height:.1f})"
        
        ax.text(center_x, center_y, label, color=color, fontsize=8, 
                ha='center', va='center', weight='bold',
                bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.7))
        
        return width < 0 or height < 0  # Return True if there are issues
    
    def draw_bbox(self, ax, coords, img_width, img_height, class_id, class_names):
        """Draw regular bounding box"""
        x_center, y_center, width, height = coords
        
        # Denormalize
        x_center *= img_width
        y_center *= img_height
        width *= img_width
        height *= img_height
        
        # Check for negative dimensions
        color = 'red' if width < 0 or height < 0 else 'lime'
        
        # Calculate top-left corner
        x = x_center - width / 2
        y = y_center - height / 2
        
        # Draw rectangle
        rect = patches.Rectangle((x, y), width, height, linewidth=2, 
                               edgecolor=color, facecolor='none')
        ax.add_patch(rect)
        
        # Add class label
        class_name = class_names[class_id] if class_id < len(class_names) else f"class_{class_id}"
        
        # Add dimension info if negative
        label = class_name
        if width < 0 or height < 0:
            label += f" (W:{width:.1f}, H:{height:.1f})"
        
        ax.text(x_center, y_center, label, color=color, fontsize=8,
                ha='center', va='center', weight='bold',
                bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.7))
        
        return width < 0 or height < 0  # Return True if there are issues
    
    def draw_rotated_bbox(self, ax, coords, img_width, img_height, class_id, class_names):
        """Draw rotated bounding box"""
        x_center, y_center, width, height, angle = coords
        
        # Denormalize
        x_center *= img_width
        y_center *= img_height
        width *= img_width
        height *= img_height
        
        # Check for negative dimensions
        color = 'red' if width < 0 or height < 0 else 'lime'
        
        # Create rotated rectangle (simplified representation)
        # For simplicity, we'll draw the bounding rectangle
        x = x_center - width / 2
        y = y_center - height / 2
        
        rect = patches.Rectangle((x, y), width, height, linewidth=2, 
                               edgecolor=color, facecolor='none', angle=angle)
        ax.add_patch(rect)
        
        # Add class label
        class_name = class_names[class_id] if class_id < len(class_names) else f"class_{class_id}"
        
        # Add dimension and angle info if negative dimensions
        label = class_name
        if width < 0 or height < 0:
            label += f" (W:{width:.1f}, H:{height:.1f}, A:{angle:.1f}°)"
        
        ax.text(x_center, y_center, label, color=color, fontsize=8,
                ha='center', va='center', weight='bold',
                bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.7))
        
        return width < 0 or height < 0  # Return True if there are issues
    
    def view_image(self, image_path, label_path, show_issues_only=False):
        """View a single image with its annotations"""
        # Load image
        if not image_path.exists():
            print(f"Image not found: {image_path}")
            return False
        
        image = cv2.imread(str(image_path))
        if image is None:
            print(f"Could not load image: {image_path}")
            return False
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img_height, img_width = image.shape[:2]
        
        # Load annotations
        annotations = self.load_annotations(label_path)
        
        # Check for issues
        has_issues = False
        for annotation in annotations:
            coords = annotation['coords']
            if annotation['format'] == 'obb' and len(coords) == 8:
                # Check OBB for negative dimensions
                denorm_coords = self.denormalize_coords(coords, img_width, img_height)
                points = [(denorm_coords[i], denorm_coords[i + 1]) for i in range(0, 8, 2)]
                x_coords = [p[0] for p in points]
                y_coords = [p[1] for p in points]
                width = max(x_coords) - min(x_coords)
                height = max(y_coords) - min(y_coords)
                if width < 0 or height < 0:
                    has_issues = True
            elif annotation['format'] in ['bbox', 'rotated_bbox']:
                # Check regular bbox for negative dimensions
                if len(coords) >= 4 and (coords[2] < 0 or coords[3] < 0):
                    has_issues = True
        
        # Skip if only showing issues and this image has none
        if show_issues_only and not has_issues:
            return False
        
        # Create figure
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        ax.imshow(image)
        
        class_names = self.dataset_info.get('names', ['object'])
        issue_count = 0
        
        # Draw annotations
        for annotation in annotations:
            coords = annotation['coords']
            class_id = annotation['class_id']
            
            if annotation['format'] == 'obb':
                if self.draw_obb(ax, coords, img_width, img_height, class_id, class_names):
                    issue_count += 1
            elif annotation['format'] == 'bbox':
                if self.draw_bbox(ax, coords, img_width, img_height, class_id, class_names):
                    issue_count += 1
            elif annotation['format'] == 'rotated_bbox':
                if self.draw_rotated_bbox(ax, coords, img_width, img_height, class_id, class_names):
                    issue_count += 1
        
        # Set title with issue info
        title = f"{image_path.name} ({len(annotations)} annotations)"
        if issue_count > 0:
            title += f" - {issue_count} ISSUES!"
        
        ax.set_title(title, fontsize=14, color='red' if issue_count > 0 else 'black')
        ax.axis('off')
        
        # Add legend
        if annotations:
            legend_elements = [
                plt.Line2D([0], [0], color='lime', lw=2, label='Normal annotations'),
                plt.Line2D([0], [0], color='red', lw=2, label='Issues (negative dimensions)')
            ]
            ax.legend(handles=legend_elements, loc='upper right')
        
        plt.tight_layout()
        plt.show()
        
        return has_issues
    
    def view_dataset(self, split='train', max_images=None, show_issues_only=False):
        """View images from a dataset split"""
        images_dir = self.dataset_path / 'images' / split
        labels_dir = self.dataset_path / 'labels' / split
        
        if not images_dir.exists():
            print(f"Images directory not found: {images_dir}")
            return
        
        if not labels_dir.exists():
            print(f"Labels directory not found: {labels_dir}")
            return
        
        # Get all image files
        image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff']
        image_files = []
        for ext in image_extensions:
            image_files.extend(images_dir.glob(ext))
            image_files.extend(images_dir.glob(ext.upper()))
        
        image_files.sort()
        
        if not image_files:
            print(f"No images found in {images_dir}")
            return
        
        print(f"Found {len(image_files)} images in {split} split")
        
        issues_found = 0
        images_shown = 0
        
        for i, image_path in enumerate(image_files):
            if max_images and images_shown >= max_images:
                break
            
            # Find corresponding label file
            label_name = image_path.stem + '.txt'
            label_path = labels_dir / label_name
            
            print(f"\\nViewing {i+1}/{len(image_files)}: {image_path.name}")
            
            if self.view_image(image_path, label_path, show_issues_only):
                issues_found += 1
            
            if not show_issues_only or self.view_image(image_path, label_path, True):
                images_shown += 1
                
                # Wait for user input
                try:
                    user_input = input("Press Enter for next image, 'q' to quit, 's' to skip to next with issues: ").strip().lower()
                    if user_input == 'q':
                        break
                    elif user_input == 's':
                        show_issues_only = True
                        print("Now showing only images with issues...")
                except KeyboardInterrupt:
                    print("\\nViewing interrupted by user")
                    break
        
        print(f"\\nSummary:")
        print(f"Total images processed: {i+1}")
        print(f"Images with issues: {issues_found}")
        if issues_found > 0:
            print(f"Issue rate: {issues_found/(i+1)*100:.1f}%")

def main():
    parser = argparse.ArgumentParser(description='View YOLO dataset with bounding boxes')
    parser.add_argument('dataset_path', help='Path to dataset directory')
    parser.add_argument('--split', choices=['train', 'val', 'test'], default='train',
                       help='Dataset split to view (default: train)')
    parser.add_argument('--max-images', type=int, help='Maximum number of images to show')
    parser.add_argument('--issues-only', action='store_true', 
                       help='Show only images with annotation issues')
    
    args = parser.parse_args()
    
    dataset_path = Path(args.dataset_path)
    if not dataset_path.exists():
        print(f"Dataset path does not exist: {dataset_path}")
        return
    
    viewer = DatasetViewer(dataset_path)
    viewer.view_dataset(args.split, args.max_images, args.issues_only)

if __name__ == "__main__":
    main()