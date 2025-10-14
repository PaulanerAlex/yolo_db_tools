import yaml
import os
import random
import shutil
from bb_tools import bb_picker


def write_to_yaml(data, filename):
    """Writes data to a YAML file with name of dataset"""
    with open(filename, 'w') as file:
        yaml.dump(data, file, default_flow_style=False)

def create_folder_structure(project_name):
    """Creates folder structure for dataset"""
    os.makedirs(f"dataset/{project_name}/images/train", exist_ok=True)
    os.makedirs(f"dataset/{project_name}/images/val", exist_ok=True)
    os.makedirs(f"dataset/{project_name}/labels/train", exist_ok=True)
    os.makedirs(f"dataset/{project_name}/labels/val", exist_ok=True)

def create_yaml_structure(project_name, object_name="object"):
    """Creates YAML structure for dataset"""
    
    yml_content = {
        'path': './',
        'train': f"./dataset/{project_name}/images/train",
        'val': f"./dataset/{project_name}/images/val",
        'names': [object_name]
    }
    return yml_content

def annotate_img(image_path, validating=False, bb_type=1, multiple=False):
    """
    Annotates image using bb_picker and saves image and txt
    in the corresponding folders
    """

    purpose = "train" if not validating else "val"

    # get bounding box coordinates by user input
    if multiple:
        annotations = bb_picker(image_path, True, rotation_enabled=(bb_type==0), multiple=True)
        if not annotations:
            print("No annotations created for this image.")
            return
    else:
        single_annotation = bb_picker(image_path, True, rotation_enabled=(bb_type==0))
        annotations = [single_annotation]  # Convert to list format for consistent processing
    
    shutil.copy(image_path, f"dataset/{project_name}/images/{purpose}/" + os.path.basename(image_path))

    # Save the coordinates to a text file
    label_path = f"dataset/{project_name}/labels/{purpose}/{os.path.basename(image_path).replace('.jpg', '.txt').replace('.png', '.txt').replace('.jpeg', '.txt')}"
    with open(label_path, 'w') as f:
        for i, annotation in enumerate(annotations):
            if multiple:
                # annotation is (corners, width, height, angle)
                [(x1, y1), (x2, y2), (x3, y3), (x4, y4)], width, height, angle = annotation
            else:
                # annotation is (corners, width, height, angle) - same format
                [(x1, y1), (x2, y2), (x3, y3), (x4, y4)], width, height, angle = annotation
            
            if bb_type == 1:  # Oriented Bounding Box
                line = f"0 {x1} {y1} {x2} {y2} {x3} {y3} {x4} {y4}"
            elif bb_type == 2:  # Rectangle format
                line = f"0 {(x1 + x3) / 2} {(y1 + y3) / 2} {width} {height}"
            
            if i > 0:  # Add newline for multiple annotations
                f.write("\n")
            f.write(line)

def dataset_tool(project_name, object_name="object", bb_type=1, multiple=False):
    """
    guide for dataset creation for the user
    """

    print(f"Creating folder structure and YAML file for {project_name} dataset...")
    create_folder_structure(project_name)

    print("how many images should be annotated for training?")

    num_train_images = int(input('>>> '))

    print("how many images should be annotated for validation?")

    num_val_images = int(input('>>> '))

    annotated_imgs = []

    rem_count = 0
    i = 0
    while i < num_train_images:
        print(f"Annotating training image {i + 1} of {num_train_images}...")
        while image_name in annotated_imgs:
            image_name = random.choice(os.listdir("img"))
        if image_name.split('.')[-1].lower() not in ['jpg', 'jpeg', 'png']:
            print(f"Skipping non-image file: {image_name} (if infinite loop occurs, please lower the number of images to annotate)")
            continue
        annotate_img(f"img/{image_name}", bb_type=bb_type, multiple=multiple)
        annotated_imgs.append(image_name)
        print("continue? (s to delete the last image from dataset, n or no to stop)")
        cont = input('>>> ')
        image_path = f"dataset/{project_name}/images/train/" + os.path.basename(image_name)
        label_path = f"dataset/{project_name}/labels/train/" + os.path.basename(image_name).replace('.jpg', '.txt').replace(".png", ".txt").replace("jpeg", "txt")

        if cont.lower() == 's':
            # remove the last image from the dataset
            if (os.path.exists(image_path)):
                os.remove(image_path)
            if (os.path.exists(label_path)):
                os.remove(label_path)
            rem_count += 1
            print("Last image removed from dataset.")
        i += 1
        if cont.lower() == 'no' or cont.lower() == 'n':
            break

    try:
        print(f"Annotated {i - rem_count} out of {num_train_images} training images.")
    except UnboundLocalError:
        print("No training images annotated.")

    rem_count = 0
    i = 0
    while i < num_val_images:
        print(f"Annotating validation image {i + 1} of {num_val_images}...")
        while image_name in annotated_imgs:
            image_name = random.choice(os.listdir("img"))
        if image_name.split('.')[-1].lower() not in ['jpg', 'jpeg', 'png']:
            print(f"Skipping non-image file: {image_name} (if infinite loop occurs, please lower the number of images to annotate)")
            continue
        annotate_img(f"img/{image_name}", validating=True, bb_type=bb_type, multiple=multiple)
        annotated_imgs.append(image_name)
        print("continue? (s to delete the last image from dataset, n or no to stop)")
        cont = input('>>> ')
        image_path = f"dataset/{project_name}/images/val/" + os.path.basename(image_name)
        label_path = f"dataset/{project_name}/labels/val/" + os.path.basename(image_name).replace('.jpg', '.txt').replace(".png", ".txt").replace("jpeg", "txt")
        if cont.lower() == 's':
            # remove the last image from the dataset
            if (os.path.exists(image_path)):
                os.remove(image_path)
            if (os.path.exists(label_path)):
                os.remove(label_path)
            rem_count += 1
            print("Last image removed from dataset.")
        i += 1
        if cont.lower() == 'no' or cont.lower() == 'n':
            break
            
    try:
        print(f"Annotated {i - rem_count} out of {num_val_images} validation images.")
    except UnboundLocalError:
        print("No validation images annotated.")

    print("Annotation complete. Creating YAML file...")


    yml_content = create_yaml_structure(project_name, object_name=object_name)
    yml_path = f"dataset/{project_name}/{project_name}.yml"
    write_to_yaml(yml_content, yml_path)

    print(f"YAML file created at {yml_path}")

if __name__ == "__main__":
    global project_name
    
    project_name = input("Enter the name of the dataset/project: ")
    if not project_name:
        raise ValueError("Project name cannot be empty.")
    
    print(f"Object names suggestions:")
    object_names = ['ellipse', 'quadrilateral', 'object']
    for i, name in enumerate(object_names):
        print(f"{i + 1}: {name}")
    object_name = input("Enter the name (or digit for suggestion) of the object to detect (default: object): ")
    if object_name.strip() == "":
        object_name = "object"
    elif object_name.isdigit() and 1 <= int(object_name) <= len(object_names):
        object_name = object_names[int(object_name) - 1]
    else:
        object_name = object_name.strip()
    print(f"Using object name: {object_name}")

    print("Select bounding box type:")
    bb_types = ['Oriented Bounding Box (4 points)', 'Rectangle (4 points)']
    for i, name in enumerate(bb_types):
        print(f"{i + 1}: {name}")
    bb_type = input("Enter bounding box type digit: ")
    if bb_type.isdigit() and (1 <= int(bb_type) <= len(bb_types)):
        bb_type = int(bb_type)
    else:
        print("Invalid input. Defaulting to 1 (Oriented Bounding Box).")
        bb_type = 1

    print(f"Using bounding box type: {bb_types[bb_type - 1]}")

    print("Do you want to allow multiple bounding boxes per image?")
    print("1: Single bounding box per image")
    print("2: Multiple bounding boxes per image")
    multiple_choice = input("Enter choice (default: 1): ").strip()
    if multiple_choice == "2":
        multiple = True
        print("Multiple bounding boxes per image enabled.")
    else:
        multiple = False
        print("Single bounding box per image mode.")

    dataset_tool(project_name, object_name=object_name, bb_type=bb_type, multiple=multiple)
