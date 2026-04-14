import json
import csv
import sys
from pathlib import Path

import numpy as np

try:
    import cv2
except ImportError:  # pragma: no cover - optional in Streamlit Cloud
    cv2 = None


def get_app_root():
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent.parent


ROOT_DIR = get_app_root()
DATA_DIR = ROOT_DIR / "data"
MODELS_DIR = ROOT_DIR / "models"
MODEL_FILE = MODELS_DIR / "face_trainer.yml"
LABELS_FILE = MODELS_DIR / "labels.json"
MIN_IMAGES_PER_PERSON = 5
EMPLOYEES_FILE = DATA_DIR / "employees.csv"


def require_cv2():
    if cv2 is None:
        raise RuntimeError(
            "OpenCV is not available in this environment. "
            "Training requires a local machine or a server with OpenCV installed."
        )
    return cv2


def preprocess_face(img):
    require_cv2()
    resized = cv2.resize(img, (200, 200))
    return cv2.equalizeHist(resized)


def load_employee_lookup():
    lookup = {}
    if not EMPLOYEES_FILE.exists():
        return lookup
    with EMPLOYEES_FILE.open("r", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            mobile = row.get("Mobile", "").strip()
            name = row.get("Name", "").strip().lower().replace(" ", "_")
            if mobile and name:
                lookup[mobile] = name
    return lookup


def train_faces():
    require_cv2()
    if not DATA_DIR.exists():
        raise FileNotFoundError("No data folder found. Run capture_faces.py first.")

    images, labels, label_map = load_training_data()
    if not images:
        raise RuntimeError("No training images found in data/.")
    if len(label_map) < 2:
        print(
            "Warning: only one employee in training data. "
            "Add more employees for reliable classification."
        )

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.train(images, labels)
    recognizer.save(str(MODEL_FILE))

    with LABELS_FILE.open("w", encoding="utf-8") as f:
        json.dump(label_map, f, indent=2)

    return MODEL_FILE, LABELS_FILE


def label_display_name(folder_name, employee_lookup):
    if "_" not in folder_name:
        return folder_name
    possible_mobile = folder_name.rsplit("_", 1)[-1]
    return employee_lookup.get(possible_mobile, folder_name)


def load_training_data():
    require_cv2()
    images = []
    labels = []
    label_map = {}
    current_label = 0
    employee_lookup = load_employee_lookup()

    for person_dir in sorted(DATA_DIR.iterdir()):
        if not person_dir.is_dir():
            continue
        if person_dir.name.startswith("test_"):
            continue

        person_images = []

        for img_path in person_dir.glob("*.jpg"):
            img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            person_images.append(preprocess_face(img))

        if len(person_images) < MIN_IMAGES_PER_PERSON:
            continue

        label_map[current_label] = label_display_name(person_dir.name, employee_lookup)
        images.extend(person_images)
        labels.extend([current_label] * len(person_images))
        current_label += 1

    return images, np.array(labels), label_map


def main():
    train_faces()


if __name__ == "__main__":
    main()
