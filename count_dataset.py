from pathlib import Path
from collections import Counter

ROOT = Path(__file__).resolve().parent

CLASS_NAMES = {
    0: "Elephant",
    1: "Tiger",
    2: "Leopard",
    3: "Bear"
}

for split in ["train", "val", "test"]:

    label_dir = (
        ROOT /
        "04_yolo_dataset" /
        "labels" /
        split
    )

    counts = Counter()

    for label_file in label_dir.glob("*.txt"):

        with open(label_file, "r") as f:

            classes_in_image = set()

            for line in f:

                parts = line.split()

                if parts:
                    classes_in_image.add(
                        int(parts[0])
                    )

            for class_id in classes_in_image:
                counts[class_id] += 1

    print()
    print("====================")
    print(split.upper())
    print("====================")

    for class_id, name in CLASS_NAMES.items():

        print(
            f"{name:10s}: "
            f"{counts[class_id]} images"
        )