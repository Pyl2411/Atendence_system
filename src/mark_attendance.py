import csv
import json
import sys
from datetime import datetime, timedelta
from pathlib import Path

import requests

try:
    import cv2
except ImportError:  # pragma: no cover - optional in Streamlit Cloud
    cv2 = None


def get_app_root():
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent.parent


ROOT_DIR = get_app_root()
MODELS_DIR = ROOT_DIR / "models"
ATTENDANCE_DIR = ROOT_DIR / "attendance"
MODEL_FILE = MODELS_DIR / "face_trainer.yml"
LABELS_FILE = MODELS_DIR / "labels.json"
CONFIDENCE_THRESHOLD = 55
MIN_WORK_HOURS = 9
ATTENDANCE_COLUMNS = [
    "Name",
    "Date",
    "CheckIn",
    "CheckInLocation",
    "CheckInLat",
    "CheckInLon",
    "CheckOut",
    "CheckOutLocation",
    "CheckOutLat",
    "CheckOutLon",
    "WorkHours",
]


def require_cv2():
    if cv2 is None:
        raise RuntimeError(
            "OpenCV is not available in this environment. "
            "Face recognition attendance requires a local machine or a server with OpenCV installed."
        )
    return cv2


def load_assets():
    require_cv2()
    if not MODEL_FILE.exists() or not LABELS_FILE.exists():
        raise FileNotFoundError("Model or labels missing. Run train_model.py first.")

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read(str(MODEL_FILE))

    with LABELS_FILE.open("r", encoding="utf-8") as f:
        labels = {int(k): v for k, v in json.load(f).items()}

    return recognizer, labels


def ensure_today_file(file_path):
    if file_path.exists():
        return
    with file_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(ATTENDANCE_COLUMNS)


def get_current_location():
    try:
        response = requests.get("http://ip-api.com/json/", timeout=3)
        response.raise_for_status()
        data = response.json()
        if data.get("status") != "success":
            return "Unknown", "", ""
        city = data.get("city", "")
        region = data.get("regionName", "")
        country = data.get("country", "")
        location = ", ".join([p for p in [city, region, country] if p]).strip()
        return location or "Unknown", str(data.get("lat", "")), str(data.get("lon", ""))
    except Exception:
        return "Unknown", "", ""


def read_rows(file_path):
    rows = []
    if not file_path.exists():
        return rows
    with file_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Backward compatibility: migrate old format (Name, Date, Time)
            # into new format (CheckIn/CheckOut/WorkHours).
            check_in = row.get("CheckIn", "").strip()
            legacy_time = row.get("Time", "").strip()
            rows.append(
                {
                    "Name": row.get("Name", "").strip(),
                    "Date": row.get("Date", "").strip(),
                    "CheckIn": check_in or legacy_time,
                    "CheckInLocation": row.get("CheckInLocation", "").strip(),
                    "CheckInLat": row.get("CheckInLat", "").strip(),
                    "CheckInLon": row.get("CheckInLon", "").strip(),
                    "CheckOut": row.get("CheckOut", "").strip(),
                    "CheckOutLocation": row.get("CheckOutLocation", "").strip(),
                    "CheckOutLat": row.get("CheckOutLat", "").strip(),
                    "CheckOutLon": row.get("CheckOutLon", "").strip(),
                    "WorkHours": row.get("WorkHours", "").strip(),
                }
            )
    return rows


def write_rows(file_path, rows):
    with file_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=ATTENDANCE_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def show_status_and_exit(frame, message):
    require_cv2()
    info_frame = frame.copy()
    cv2.putText(
        info_frame,
        message,
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 255),
        2,
    )
    cv2.imshow("Attendance - Face Recognition", info_frame)
    cv2.waitKey(2500)


def preprocess_face(face_roi):
    require_cv2()
    resized = cv2.resize(face_roi, (200, 200))
    return cv2.equalizeHist(resized)


def start_attendance():
    require_cv2()
    recognizer, labels = load_assets()
    ATTENDANCE_DIR.mkdir(parents=True, exist_ok=True)

    today_file = ATTENDANCE_DIR / f"attendance_{datetime.now().strftime('%Y%m%d')}.csv"
    ensure_today_file(today_file)

    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        raise RuntimeError("Could not open camera.")

    print("Press 'q' to quit.")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                continue

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5)

            for (x, y, w, h) in faces:
                face_roi = gray[y : y + h, x : x + w]
                processed_face = preprocess_face(face_roi)
                label_id, confidence = recognizer.predict(processed_face)

                if confidence < CONFIDENCE_THRESHOLD:
                    name = labels.get(label_id, "unknown")
                    color = (0, 220, 0)
                    status_message, is_marked = mark_attendance(name, today_file)
                else:
                    name = "unknown"
                    color = (0, 0, 255)
                    status_message, is_marked = ("Face not recognized.", False)

                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                cv2.putText(
                    frame,
                    f"{name} ({confidence:.1f})",
                    (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    color,
                    2,
                )

                # Show clear user feedback and auto close when attendance is marked.
                cv2.putText(
                    frame,
                    status_message,
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (0, 255, 255),
                    2,
                )
                if is_marked:
                    show_status_and_exit(frame, status_message)
                    return

            cv2.imshow("Attendance - Face Recognition", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


def mark_attendance(name, file_path):
    rows = read_rows(file_path)
    now = datetime.now()
    today = now.strftime("%Y-%m-%d")
    now_time = now.strftime("%H:%M:%S")
    location, lat, lon = get_current_location()

    # Single row per employee per day: first recognition is CheckIn, second is CheckOut.
    for row in rows:
        if row.get("Name") != name or row.get("Date") != today:
            continue

        check_in = row.get("CheckIn", "").strip()
        check_out = row.get("CheckOut", "").strip()

        if check_in and check_out:
            return f"{name}: already marked IN and OUT today.", True

        if check_in and not check_out:
            check_in_dt = datetime.strptime(f"{today} {check_in}", "%Y-%m-%d %H:%M:%S")
            worked = now - check_in_dt
            required = timedelta(hours=MIN_WORK_HOURS)

            if worked < required:
                remaining = required - worked
                hours = int(remaining.total_seconds() // 3600)
                minutes = int((remaining.total_seconds() % 3600) // 60)
                return (
                    f"{name}: OUT not allowed yet. Remaining {hours}h {minutes}m.",
                    False,
                )

            row["CheckOut"] = now_time
            row["CheckOutLocation"] = location
            row["CheckOutLat"] = lat
            row["CheckOutLon"] = lon
            row["WorkHours"] = f"{worked.total_seconds() / 3600:.2f}"
            write_rows(file_path, rows)
            print(f"OUT marked: {name} at {now_time} | Location: {location} ({lat}, {lon})")
            return f"{name}: OUT at {now_time} | {location}", True

    rows.append(
        {
            "Name": name,
            "Date": today,
            "CheckIn": now_time,
            "CheckInLocation": location,
            "CheckInLat": lat,
            "CheckInLon": lon,
            "CheckOut": "",
            "CheckOutLocation": "",
            "CheckOutLat": "",
            "CheckOutLon": "",
            "WorkHours": "",
        }
    )
    write_rows(file_path, rows)
    print(f"IN marked: {name} at {now_time} | Location: {location} ({lat}, {lon})")
    return f"{name}: IN at {now_time} | {location}", True


def main():
    start_attendance()


if __name__ == "__main__":
    main()
