import os
import random
import numpy as np
from PIL import Image
from imgaug import augmenters as iaa


def get_augmentation_pipeline():
    return iaa.Sequential([
        iaa.Fliplr(0.5),                        # Random horizontal flip
        iaa.Affine(rotate=(-20, 20)),           # Random rotation between -20° and 20°
        iaa.Multiply((0.8, 1.2)),               # Random brightness (multiply pixel values)
        iaa.GaussianBlur(sigma=(0.0, 1.0)),     # Random blur with varying intensity
        iaa.AdditiveGaussianNoise(scale=(0, 0.05*255)), # Add Gaussian noise to pixels
    ])


def copy_existing_images(class_path, output_class_path):
    """
    Copy existing images from the original class directory to the output directory.
    """
    for img_file in os.listdir(class_path):
        img_src = os.path.join(class_path, img_file)
        img_dst = os.path.join(output_class_path, img_file)
        if not os.path.exists(img_dst):
            Image.open(img_src).save(img_dst)


def augment_class(augmentation, class_dir, output_class_dir, target_count):
    """
    Augment images for one class directory.
    """
    image_paths = [
        os.path.join(class_dir, f)
        for f in os.listdir(class_dir)
        if f.lower().endswith(('png', 'jpg', 'jpeg'))
    ]
    existing_count = len(image_paths)
    print(f"Processing {class_dir}, {existing_count} images")

    if existing_count >= target_count:
        print(f"Skipping {class_dir}, already has {existing_count} images.")
        return

    count = 0
    while existing_count + count < target_count:
        image_path = random.choice(image_paths)
        image = Image.open(image_path).convert('RGB')
        image_np = np.array(image)
        augmented_image_np = augmentation(image=image_np)
        augmented_image = Image.fromarray(augmented_image_np)

        save_name = f"{os.path.basename(image_path).split('.')[0]}_aug_{count}.png"
        augmented_image.save(os.path.join(output_class_dir, save_name))
        count += 1

    print(f"Generated {count} new images for {class_dir}\n")


def augment_data(augmentation, input_dir, output_dir, target_count):
    """
    Augment images for all class directory.
    """
    # Iterate through each class directory
    for class_name in os.listdir(input_dir):
        class_path = os.path.join(input_dir, class_name)
        output_class_path = os.path.join(output_dir, class_name)

        if os.path.isdir(class_path):
            os.makedirs(output_class_path, exist_ok=True)
            copy_existing_images(class_path, output_class_path)
            augment_class(
                augmentation,
                class_path,
                output_class_path,
                target_count
            )


if __name__ == "__main__":
    input_dir = 'images'  # Directory containing class folders
    output_dir = 'augmented_images'  # Directory to save augmented images
    target_count = 464  # Target number of images per class

    augmentation = get_augmentation_pipeline()

    augment_data(augmentation, input_dir, output_dir, target_count)

    print("Done augmenting all classes.")
