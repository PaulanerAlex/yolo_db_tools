from ultralytics import YOLO
import os
import pathlib

models_avail = ["yolo11s-obb.pt", "yolo11s.pt"]

print("--- MODEL SELECTION ---")
print(f"Available models:")
for i, name in enumerate(models_avail):
    print(f"{i+1}: {name}")

input("Which model do you want to use? (type in number, press Enter for default yolo11s.pt): ")

if input.strip().isdigit() and 1 <= int(input.strip()) <= len(models_avail):
    model_name = models_avail[int(input.strip()) - 1]
else:
    model_name = "yolo11s.pt"

print(f"Using model: {model_name}")

print("\n--- DATASET SELECTION ---")

dataset_avail = os.listdir("dataset")
if not dataset_avail:
    raise ValueError("No datasets found in the 'dataset' directory.")

print(f"Available datasets:")
for i, name in enumerate(dataset_avail):
    print(f"{i+1}: {name}")

input("Which dataset do you want to use? (type in number, press Enter for default ellipse_recognition): ")

if input.strip().isdigit() and 1 <= int(input.strip()) <= len(dataset_avail):
    dataset_name = dataset_avail[int(input.strip()) - 1]
else:
    dataset_name = "ellipse_recognition"

print(f"Using dataset: {dataset_name}\n")

dataset_yml_path = pathlib.Path("dataset", dataset_name, f"{dataset_name}.yml")

if not dataset_yml_path.exists():
    raise FileNotFoundError(f"Dataset YAML file not found at {dataset_yml_path}")

print("\n--- CONFIG SELECTION ---")

# TODO: add config selection
print("Using default config settings. You can modify the training parameters in the script if needed.\n")

if __name__ == "__main__":
    # Load model
    model = YOLO(model_name)
    # Train the model
    results = model.train(data=dataset_yml_path, epochs=250, imgsz=640)

# model = YOLO("runs/obb/train/weights/best.pt") # Load the trained model

# results = model("test4.jpg", show=True) # predict sample image

# # Access the results
# for result in results:
#     xywhr = result.keypoints.xy  # center-x, center-y, width, height, angle (radians)
#     xyxyxyxy = result.obb.xyxyxyxy  # polygon format with 4-points
#     names = [result.names[cls.item()] for cls in result.obb.cls.int()]  # class name of each box
#     confs = result.obb.conf  # confidence score of each box

