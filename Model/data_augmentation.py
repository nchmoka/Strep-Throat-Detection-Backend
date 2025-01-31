import os
import random
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
from sklearn.model_selection import train_test_split

# Set random seeds for reproducibility
random.seed(42)
np.random.seed(42)

# Parameters
IMG_SIZE = (224, 224)
AUGMENTATIONS_PER_IMAGE = 5
VALIDATION_SPLIT = 0.15
TEST_SPLIT = 0.15

def apply_gaussian_noise(img, mean=0, sigma=25):
    """Add Gaussian noise to the image."""
    img_array = np.array(img)
    noise = np.random.normal(mean, sigma, img_array.shape)
    noisy_img = np.clip(img_array + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(noisy_img)

def apply_gaussian_blur(img, radius=2):
    """Apply Gaussian blur."""
    return img.filter(ImageFilter.GaussianBlur(radius=radius))

def adjust_contrast(img, factor=None):
    """Adjust image contrast."""
    if factor is None:
        factor = random.uniform(0.8, 1.2)
    enhancer = ImageEnhance.Contrast(img)
    return enhancer.enhance(factor)

def adjust_sharpness(img, factor=None):
    """Adjust image sharpness."""
    if factor is None:
        factor = random.uniform(0.8, 1.5)
    enhancer = ImageEnhance.Sharpness(img)
    return enhancer.enhance(factor)

def augment_image(img):
    """Apply random augmentations with varying probabilities."""
    # Random horizontal flip
    if random.random() < 0.5:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)

    # Random rotation
    angle = random.randint(-30, 30)
    img = img.rotate(angle, resample=Image.Resampling.BILINEAR, expand=False)

    # Random brightness
    brightness_factor = random.uniform(0.8, 1.2)
    img = ImageEnhance.Brightness(img).enhance(brightness_factor)

    # Additional random augmentations
    if random.random() < 0.3:
        img = apply_gaussian_noise(img, sigma=random.uniform(10, 30))

    if random.random() < 0.3:
        img = apply_gaussian_blur(img, radius=random.uniform(0.5, 1.5))

    if random.random() < 0.4:
        img = adjust_contrast(img)

    if random.random() < 0.3:
        img = adjust_sharpness(img)

    return img

def balance_classes(all_images):
    """
    Oversample the minority class so that each class has the same number of samples.
    all_images: list of (img_path, class_label) pairs.
    """
    # Separate by class
    healthy_images = [item for item in all_images if item[1] == "healthy"]
    strep_images   = [item for item in all_images if item[1] == "strep"]

    # Determine which is smaller
    len_healthy = len(healthy_images)
    len_strep   = len(strep_images)

    if len_healthy > len_strep:
        # Need to oversample strep
        while len(strep_images) < len_healthy:
            strep_images.append(random.choice(strep_images))
    elif len_strep > len_healthy:
        # Need to oversample healthy
        while len(healthy_images) < len_strep:
            healthy_images.append(random.choice(healthy_images))
    # else: they are already balanced

    # Combine and shuffle
    balanced = healthy_images + strep_images
    random.shuffle(balanced)
    return balanced

def process_dataset(input_dir, output_base):
    """Process and augment the dataset."""

    # Create output directories
    splits = ['train', 'val', 'test']
    classes = ['healthy', 'strep']

    for split in splits:
        for cls in classes:
            os.makedirs(os.path.join(output_base, split, cls), exist_ok=True)

    # Collect all images
    all_images = []
    for cls in classes:
        cls_dir = os.path.join(input_dir, cls)
        for img_name in os.listdir(cls_dir):
            if img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                img_path = os.path.join(cls_dir, img_name)
                all_images.append((img_path, cls))

    # -----------------------------
    # 1) Balance the two classes
    # -----------------------------
    print("Balancing the dataset (oversampling the smaller class)...")
    balanced_images = balance_classes(all_images)

    # -----------------------------
    # 2) Split into train/val/test
    # -----------------------------
    train_val, test = train_test_split(
        balanced_images,
        test_size=TEST_SPLIT,
        stratify=[c for _, c in balanced_images],
        random_state=42
    )
    train, val = train_test_split(
        train_val,
        test_size=VALIDATION_SPLIT/(1 - TEST_SPLIT),
        stratify=[c for _, c in train_val],
        random_state=42
    )

    def process_images(dataset, split_name, augment=False):
        """Process and save images for a specific split."""
        for img_path, cls_label in dataset:
            img = Image.open(img_path).convert('RGB')
            img = img.resize(IMG_SIZE, Image.Resampling.LANCZOS)

            # Save original image
            base_name = os.path.splitext(os.path.basename(img_path))[0]
            save_dir = os.path.join(output_base, split_name, cls_label)
            original_path = os.path.join(save_dir, f"{base_name}_orig.jpg")
            img.save(original_path, 'JPEG', quality=95)

            # Augment only if specified (typically for 'train')
            if augment:
                for i in range(AUGMENTATIONS_PER_IMAGE):
                    aug_img = augment_image(img)
                    aug_path = os.path.join(save_dir, f"{base_name}_aug{i}.jpg")
                    aug_img.save(aug_path, 'JPEG', quality=95)

    print("Processing training set...")
    process_images(train, 'train', augment=True)

    print("Processing validation set...")
    process_images(val, 'val', augment=False)

    print("Processing test set...")
    process_images(test, 'test', augment=False)

    # Print dataset statistics
    print("\nDataset Statistics:")
    print(f"Total original images: {len(all_images)}")
    print(f"After balancing: {len(balanced_images)}")
    print(f"Training set: {len(train)} (each with {AUGMENTATIONS_PER_IMAGE} augmentations = {len(train)*(AUGMENTATIONS_PER_IMAGE+1)} total)")
    print(f"Validation set: {len(val)}")
    print(f"Test set: {len(test)}")

if __name__ == "__main__":
    # Run the data augmentation pipeline
    process_dataset(
        input_dir='data',         # e.g. data/healthy, data/strep
        output_base='processed_data'
    )
