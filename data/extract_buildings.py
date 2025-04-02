import json
import os
import numpy as np
import cv2
from tqdm import tqdm


def create_output_dirs(base_dir, num_classes=6):
    """Create output directories for each damage level"""
    for i in range(num_classes):
        os.makedirs(os.path.join(base_dir, f"ds{i}"), exist_ok=True)


def load_mask(mask_path):
    """
    Load a mask image and return a binary version (0 and 1) and the grayscale mask
    """
    mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
    if mask is None:
        raise FileNotFoundError(f"Mask not found: {mask_path}")

    # Use the first channel if the mask is 3D, otherwise use it directly
    mask_gray = mask[:, :, 0] if mask.ndim == 3 else mask
    return mask_gray


def extract_and_save_building(image_path, mask_path, output_path):
    """
    Apply the mask, crop the region around the object, and save the result
    """
    image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if image is None:
        raise FileNotFoundError(f"Image not found: {image_path}")

    mask_gray = load_mask(mask_path)

    # Apply the mask to the image
    masked_image = cv2.bitwise_and(image, image, mask=mask_gray)

    # Find bounding box of the masked region
    coords = np.argwhere(mask_gray == 255)
    if coords.size == 0:
        print(f"Warning: No object found in mask {mask_path}")
        return

    y0, x0 = coords.min(axis=0) - 3
    y1, x1 = coords.max(axis=0) + 3

    # Clamp bounding box coordinates to image dimensions
    H, W = masked_image.shape[:2]
    x0, y0 = max(x0, 0), max(y0, 0)
    x1, y1 = min(x1, W), min(y1, H)

    # Crop the masked image to the bounding box
    image_cropped = masked_image[y0:y1, x0:x1]
    if image_cropped.size == 0:
        print(f"Warning: Cropped image is empty for {image_path}")
        return

    # Save the cropped image
    cv2.imwrite(output_path, image_cropped)


def process_json_files(input_base_dir, output_base_dir, num_classes=6):
    """
    Process all JSON files and extract/save building crops from image/mask pairs
    """
    json_dir = os.path.join(input_base_dir, "JSON")
    json_files = os.listdir(json_dir)
    data = {}
    ds_counters = {str(i): 0 for i in range(num_classes)}

    for file_name in tqdm(json_files, desc="Processing JSON files"):
        with open(os.path.join(json_dir, file_name)) as f:
            file_data = json.load(f)

        for building in file_data['Buildings']:
            damage_level = str(building[3])
            ds_counters[damage_level] += 1

            image_name = f"ds{damage_level}_{ds_counters[damage_level]:03d}"
            output_path = os.path.join(output_base_dir, f"ds{damage_level}", f"{image_name}.png")

            frame_path = os.path.join(input_base_dir, file_data['Frame_Name'])
            mask_path = os.path.join(input_base_dir, "MASK", building[2])

            # Extract and save the masked and cropped building image
            extract_and_save_building(frame_path, mask_path, output_path)

            # Store metadata for this image
            building_data = {
                "image_path": output_path,
                "features": {
                    "frame_path": frame_path,
                    "mask_path": mask_path,
                    "building_id": building[0],
                    "global_coordinate": building[1],
                    "damage_level": damage_level,
                    "num_stories": str(building[4]),
                    "annotat_effort": str(building[5])
                }
            }
            data[image_name] = building_data

    return data


if __name__ == "__main__":
    input_dir = "./DoriaNET/"
    output_dir = "images"
    num_classes = 6

    # Create necessary output directories
    create_output_dirs(output_dir)

    # Process all JSON files and extract buildings
    dataset = process_json_files(input_dir, output_dir, num_classes)

    # Save metadata dictionary to a single JSON file
    with open("./images.json", 'w') as out_file:
        json.dump(dataset, out_file, indent=4)

    print("All buildings processed and data saved.")