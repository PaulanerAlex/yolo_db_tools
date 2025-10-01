#!/usr/bin/env python3
"""
Simple script to remove the last value from YOLO annotation files.
Useful when converting from oriented bounding boxes (OBB) to regular bounding boxes.
"""

import os
import glob

def fix_annotation_file(file_path):
    """Remove the last value from each line in a YOLO annotation file."""
    try:
        with open(file_path, 'r') as f:
            lines = f.readlines()
        
        fixed_lines = []
        for line in lines:
            line = line.strip()
            if line:  # Skip empty lines
                parts = line.split()
                if len(parts) > 1:  # Ensure there's something to remove
                    # Remove the last value and rejoin
                    fixed_line = ' '.join(parts[:-1])
                    fixed_lines.append(fixed_line + '\n')
                else:
                    # Keep the line as is if it's too short
                    fixed_lines.append(line + '\n')
        
        # Write back to file
        with open(file_path, 'w') as f:
            f.writelines(fixed_lines)
        
        print(f"Fixed: {file_path}")
        return True
        
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return False

def main():
    # Find all .txt files in current directory
    txt_files = glob.glob("*.txt")
    
    if not txt_files:
        print("No .txt files found in current directory.")
        return
    
    print(f"Found {len(txt_files)} annotation files.")
    
    # Ask for confirmation
    response = input("Do you want to remove the last value from each annotation? (y/n): ")
    if response.lower() != 'y':
        print("Operation cancelled.")
        return
    
    # Process each file
    success_count = 0
    for txt_file in txt_files:
        if fix_annotation_file(txt_file):
            success_count += 1
    
    print(f"\nProcessed {success_count}/{len(txt_files)} files successfully.")

if __name__ == "__main__":
    main()