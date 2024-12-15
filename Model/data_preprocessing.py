import os
import random
from PIL import Image, ImageEnhance
from sklearn.model_selection import train_test_split

# Random seed for reproducibility
random.seed(42)

# Input directories
input_dir = 'data'
classes = ['healthy', 'strep']

# Output directory
output_base = 'processed_data'
os.makedirs(output_base, exist_ok=True)

splits = ['train', 'val', 'test']
for sp in splits:
    for cls in classes:
        os.makedirs(os.path.join(output_base, sp, cls), exist_ok=True)

# Collect image paths
all_images = []
for cls in classes:
    cls_dir = os.path.join(input_dir, cls)
    for img_name in os.listdir(cls_dir):
        if img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
            all_images.append((os.path.join(cls_dir, img_name), cls))


# Split sizes
# Data: healthy=215, strep=147 total=362
# Train: 70% of 362 ~ 253 images
# Val: 15% of 362 ~ 54 images
# Test: 15% of 362 ~ 55 images

train_val, test = train_test_split(all_images, test_size=0.15, stratify=[c for _, c in all_images], random_state=42)
train, val = train_test_split(train_val, test_size=0.1765, stratify=[c for _, c in train_val], random_state=42)

print(f"Total images: {len(all_images)}")
print(f"Train: {len(train)}")
print(f"Val: {len(val)}")
print(f"Test: {len(test)}")

# Parameters
IMG_SIZE = (224, 224)
AUGMENTATIONS_PER_IMAGE = 3  # how many augmented images to produce per training image

def preprocess_image(img_path):
    """ Load and preprocess a single image. """
    img = Image.open(img_path).convert('RGB')
    img = img.resize(IMG_SIZE)
    return img

def save_image(img, path):
    """ Save image in JPEG format. """
    img.save(path, 'JPEG', quality=95)

def augment_image(img):
    """Apply random augmentations to the image and return a new image."""
    # Random rotation: -30 to 30 degrees
    angle = random.randint(-30, 30)
    img_aug = img.rotate(angle)

    # Random horizontal flip with 50% chance
    if random.random() < 0.5:
        img_aug = img_aug.transpose(Image.FLIP_LEFT_RIGHT)

    # Random brightness change
    enhancer = ImageEnhance.Brightness(img_aug)
    brightness_factor = 0.8 + 0.4 * random.random()  # from 0.8 to 1.2
    img_aug = enhancer.enhance(brightness_factor)

    return img_aug

def process_and_save_dataset(dataset, split, augment=False):
    """
    Preprocess and save images to the given split folder.
    If augment is True, apply augmentation to the training set.
    """
    for idx, (img_path, cls) in enumerate(dataset):
        img = preprocess_image(img_path)

        base_name = os.path.splitext(os.path.basename(img_path))[0]
        save_dir = os.path.join(output_base, split, cls)

        # Save original preprocessed image
        original_path = os.path.join(save_dir, f"{base_name}_orig.jpg")
        save_image(img, original_path)

        # If training, produce augmented images
        if augment:
            for i in range(AUGMENTATIONS_PER_IMAGE):
                img_aug = augment_image(img)
                aug_path = os.path.join(save_dir, f"{base_name}_aug{i}.jpg")
                save_image(img_aug, aug_path)

# Process the datasets
process_and_save_dataset(train, 'train', augment=True)
process_and_save_dataset(val, 'val', augment=False)
process_and_save_dataset(test, 'test', augment=False)

print("Data preprocessing, augmentation, and splitting completed!")
