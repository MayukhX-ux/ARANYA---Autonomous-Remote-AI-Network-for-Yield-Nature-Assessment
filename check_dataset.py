import cv2
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parent

IMAGE_DIR = ROOT / "04_yolo_dataset" / "images" / "train"
LABEL_DIR = ROOT / "04_yolo_dataset" / "labels" / "train"

CLASS_NAMES = {
    0: "Elephant",
    1: "Tiger",
    2: "Leopard",
    3: "Bear"
}

images = list(IMAGE_DIR.glob("*"))

random.shuffle(images)

# Check 30 random images
images = images[:30]

for image_path in images:

    image = cv2.imread(str(image_path))

    if image is None:
        continue

    h, w = image.shape[:2]

    label_path = LABEL_DIR / (image_path.stem + ".txt")

    if label_path.exists():

        with open(label_path, "r") as f:

            for line in f:

                parts = line.strip().split()

                if len(parts) != 5:
                    continue

                class_id = int(parts[0])

                x_center = float(parts[1])
                y_center = float(parts[2])
                box_width = float(parts[3])
                box_height = float(parts[4])

                x1 = int((x_center - box_width / 2) * w)
                y1 = int((y_center - box_height / 2) * h)

                x2 = int((x_center + box_width / 2) * w)
                y2 = int((y_center + box_height / 2) * h)

                name = CLASS_NAMES.get(
                    class_id,
                    "Unknown"
                )

                cv2.rectangle(
                    image,
                    (x1, y1),
                    (x2, y2),
                    (0, 255, 0),
                    2
                )

                cv2.putText(
                    image,
                    name,
                    (x1, max(y1 - 10, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2
                )

    cv2.imshow(
        f"ARANYA Dataset - {image_path.name}",
        image
    )

    print(
        "Showing:",
        image_path.name
    )

    print(
        "Press any key for next image."
    )

    print(
        "Press Q to stop."
    )

    key = cv2.waitKey(0)

    if key == ord("q"):
        break

cv2.destroyAllWindows()