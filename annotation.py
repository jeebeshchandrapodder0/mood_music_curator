import cv2
from ultralytics import YOLO
import os

# Load YOLOv8 model for face detection (you can choose different versions like yolov8n, yolov8s, etc.)
model = YOLO('yolov8n.pt')  # You can use the smaller or larger model depending on your resources

# Specify the dataset folder paths
images_folder = "src/moodMusicCurator/dataset/WIDER_train/images"
labels_folder = "src/moodMusicCurator/dataset/WIDER_train/labels"

# Create the labels folder if not already created
os.makedirs(labels_folder, exist_ok=True)

# Iterate over images in the folder
for image_name in os.listdir(images_folder):
    if image_name.endswith(('.jpg', '.jpeg', '.png')):
        img_path = os.path.join(images_folder, image_name)
        img = cv2.imread(img_path)

        # Use YOLOv8 to detect faces
        results = model(img)

        # Extract bounding box results
        faces = results[0].boxes.xyxy.cpu().numpy()  # Face bounding box coordinates

        # Prepare the label file path (convert .jpg to .txt)
        label_file_path = os.path.join(labels_folder, image_name.replace(".jpg", ".txt").replace(".png", ".txt"))
        
        with open(label_file_path, 'w') as label_file:
            for (x1, y1, x2, y2) in faces:
                # Normalize coordinates as per YOLO format
                x_center = (x1 + x2) / 2 / img.shape[1]
                y_center = (y1 + y2) / 2 / img.shape[0]
                width = (x2 - x1) / img.shape[1]
                height = (y2 - y1) / img.shape[0]

                # Write the label in YOLO format (class_id x_center y_center width height)
                label_file.write(f"0 {x_center} {y_center} {width} {height}\n")

        print(f"Processed {image_name}, saved labels to {label_file_path}")
