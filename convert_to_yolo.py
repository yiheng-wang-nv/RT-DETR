import json
import os

# Path to the COCO JSON file
coco_json_path = '/raid/colon_reproduce/real-colon-dataset/real_colon_dataset_coco_fmt_3subsets_poslesion1000_negratio0/test_ann.json'

# Directory to save the YOLO format labels
output_dir = '/raid/colon_reproduce/real-colon-dataset/real_colon_dataset_coco_fmt_3subsets_poslesion1000_negratio0/test_labels'
os.makedirs(output_dir, exist_ok=True)

# Load the COCO JSON file
with open(coco_json_path, 'r') as f:
    coco_data = json.load(f)

# Extract images and annotations
images = {img['id']: img for img in coco_data['images']}
annotations = coco_data['annotations']

# Convert COCO annotations to YOLO format
for ann in annotations:
    image_id = ann['image_id']
    image_info = images[image_id]
    img_width = image_info['width']
    img_height = image_info['height']
    
    # COCO bounding box format: [x, y, width, height]
    x, y, width, height = ann['bbox']
    
    # Convert to YOLO format
    x_center = (x + width / 2) / img_width
    y_center = (y + height / 2) / img_height
    width /= img_width
    height /= img_height
    
    # Set the class ID to 0 for YOLO format
    class_id = 0
    
    # Prepare the YOLO annotation line
    yolo_line = f"{class_id} {x_center} {y_center} {width} {height}\n"
    
    # Save to the corresponding label file
    label_file_path = os.path.join(output_dir, f"{os.path.splitext(image_info['file_name'])[0]}.txt")
    with open(label_file_path, 'a') as label_file:
        label_file.write(yolo_line)

print("Conversion completed!")
