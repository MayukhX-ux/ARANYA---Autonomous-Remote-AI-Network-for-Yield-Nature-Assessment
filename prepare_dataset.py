from pathlib import Path
import shutil
import random

# ==========================================================
# ARANYA VISION DATASET BUILDER
# ==========================================================

ROOT = Path(__file__).resolve().parent

RAW_DIR = ROOT / "01_raw_datasets"
NEGATIVE_DIR = ROOT / "02_negative_images" / "images"
OUTPUT_DIR = ROOT / "04_yolo_dataset"

# Final ARANYA class IDs
CLASS_MAPPING = {
    "elephant": 0,
    "tiger": 1,
    "leopard": 2,
    "bear": 3
}

VALID_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".webp"
}

# Reproducible split
random.seed(42)


# ==========================================================
# DELETE OLD OUTPUT AND CREATE NEW FOLDERS
# ==========================================================

if OUTPUT_DIR.exists():
    print("Deleting old 04_yolo_dataset...")
    shutil.rmtree(OUTPUT_DIR)

for split in ["train", "val", "test"]:

    (OUTPUT_DIR / "images" / split).mkdir(
        parents=True,
        exist_ok=True
    )

    (OUTPUT_DIR / "labels" / split).mkdir(
        parents=True,
        exist_ok=True
    )


# ==========================================================
# SPLIT FUNCTION
# 70% TRAIN
# 20% VALIDATION
# 10% TEST
# ==========================================================

def split_data(items):

    random.shuffle(items)

    total = len(items)

    train_end = int(total * 0.70)
    val_end = train_end + int(total * 0.20)

    train = items[:train_end]
    val = items[train_end:val_end]
    test = items[val_end:]

    return train, val, test


# ==========================================================
# COPY POSITIVE ANIMAL DATA
# ==========================================================

def copy_animal_data(items, animal, class_id, split):

    for index, item in enumerate(items):

        image_path = item["image"]
        label_path = item["label"]

        extension = image_path.suffix.lower()

        new_stem = f"{animal}_{split}_{index:05d}"

        new_image = (
            OUTPUT_DIR /
            "images" /
            split /
            f"{new_stem}{extension}"
        )

        new_label = (
            OUTPUT_DIR /
            "labels" /
            split /
            f"{new_stem}.txt"
        )

        # Copy image
        shutil.copy2(
            image_path,
            new_image
        )

        # Read original annotation
        with open(
            label_path,
            "r",
            encoding="utf-8"
        ) as f:

            original_lines = f.readlines()

        converted_lines = []

        for line in original_lines:

            parts = line.strip().split()

            if len(parts) != 5:
                continue

            # Since each source dataset is confirmed
            # single-class, replace its original class ID
            # with ARANYA's class ID.

            _, x, y, w, h = parts

            converted_lines.append(
                f"{class_id} {x} {y} {w} {h}\n"
            )

        with open(
            new_label,
            "w",
            encoding="utf-8"
        ) as f:

            f.writelines(converted_lines)


# ==========================================================
# PROCESS EACH ANIMAL SEPARATELY
# ==========================================================

statistics = {}

for animal, class_id in CLASS_MAPPING.items():

    print()
    print("=" * 50)
    print(f"PROCESSING {animal.upper()}")
    print("=" * 50)

    animal_folder = RAW_DIR / animal

    items = []

    # Collect ALL images from original Roboflow splits
    for original_split in ["train", "valid", "test"]:

        image_folder = (
            animal_folder /
            original_split /
            "images"
        )

        label_folder = (
            animal_folder /
            original_split /
            "labels"
        )

        if not image_folder.exists():
            continue

        for image_path in image_folder.iterdir():

            if image_path.suffix.lower() not in VALID_EXTENSIONS:
                continue

            label_path = (
                label_folder /
                f"{image_path.stem}.txt"
            )

            if not label_path.exists():

                print(
                    "Missing label:",
                    image_path.name
                )

                continue

            items.append({
                "image": image_path,
                "label": label_path
            })

    print(
        f"Total {animal} images found:",
        len(items)
    )

    train_items, val_items, test_items = split_data(items)

    copy_animal_data(
        train_items,
        animal,
        class_id,
        "train"
    )

    copy_animal_data(
        val_items,
        animal,
        class_id,
        "val"
    )

    copy_animal_data(
        test_items,
        animal,
        class_id,
        "test"
    )

    statistics[animal] = {
        "total": len(items),
        "train": len(train_items),
        "val": len(val_items),
        "test": len(test_items)
    }


# ==========================================================
# PROCESS NEGATIVE FOREST IMAGES
# ==========================================================

print()
print("=" * 50)
print("PROCESSING NEGATIVE FOREST IMAGES")
print("=" * 50)

negative_images = []

if NEGATIVE_DIR.exists():

    for image_path in NEGATIVE_DIR.iterdir():

        if image_path.suffix.lower() in VALID_EXTENSIONS:

            negative_images.append(image_path)


print(
    "Negative images found:",
    len(negative_images)
)

neg_train, neg_val, neg_test = split_data(
    negative_images
)


def copy_negatives(items, split):

    for index, image_path in enumerate(items):

        extension = image_path.suffix.lower()

        new_stem = (
            f"negative_{split}_{index:05d}"
        )

        destination_image = (
            OUTPUT_DIR /
            "images" /
            split /
            f"{new_stem}{extension}"
        )

        destination_label = (
            OUTPUT_DIR /
            "labels" /
            split /
            f"{new_stem}.txt"
        )

        shutil.copy2(
            image_path,
            destination_image
        )

        # IMPORTANT:
        # Empty label = no target object in image

        destination_label.touch()


copy_negatives(
    neg_train,
    "train"
)

copy_negatives(
    neg_val,
    "val"
)

copy_negatives(
    neg_test,
    "test"
)


statistics["negative"] = {
    "total": len(negative_images),
    "train": len(neg_train),
    "val": len(neg_val),
    "test": len(neg_test)
}


# ==========================================================
# CREATE DATA.YAML
# ==========================================================

yaml_content = """path: .
train: images/train
val: images/val
test: images/test

names:
  0: elephant
  1: tiger
  2: leopard
  3: bear
"""

with open(
    OUTPUT_DIR / "data.yaml",
    "w",
    encoding="utf-8"
) as f:

    f.write(yaml_content)


# ==========================================================
# PRINT SUMMARY
# ==========================================================

print()
print()
print("=" * 65)
print("ARANYA DATASET COMPLETE")
print("=" * 65)

print(
    f"{'CLASS':<15}"
    f"{'TOTAL':>10}"
    f"{'TRAIN':>10}"
    f"{'VAL':>10}"
    f"{'TEST':>10}"
)

print("-" * 65)

for name, values in statistics.items():

    print(
        f"{name:<15}"
        f"{values['total']:>10}"
        f"{values['train']:>10}"
        f"{values['val']:>10}"
        f"{values['test']:>10}"
    )

print("-" * 65)

total_images = sum(
    values["total"]
    for values in statistics.values()
)

print(
    f"{'TOTAL':<15}"
    f"{total_images:>10}"
)

print()
print("CLASS IDs:")
print("0 = Elephant")
print("1 = Tiger")
print("2 = Leopard")
print("3 = Bear")

print()
print("Dataset saved to:")
print(OUTPUT_DIR)

print()
print("NEXT STEP: Verify annotations before training.")