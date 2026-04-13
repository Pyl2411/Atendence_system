import io
import csv
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image
from streamlit.components.v1 import html

try:
    import cv2
except ImportError:  # pragma: no cover - optional in Streamlit Cloud
    cv2 = None

from src.capture_faces import append_capture_log, sanitize_mobile, sanitize_name, upsert_employee
from src.data_store import ATTENDANCE_DIR, DATA_DIR, ROOT_DIR, load_attendance_records, load_employees
from src.mark_attendance import ATTENDANCE_COLUMNS, CONFIDENCE_THRESHOLD, mark_attendance
from src.train_model import LABELS_FILE, MODEL_FILE, MODELS_DIR, load_training_data, train_faces


APP_TITLE = "Attendance Web App"
DEFAULT_COMPANY_NAME = "Vickhardth Automation"
ROLES = ["Manager", "Co Founder", "Employee", "Team Leader", "Trainee"]
CV2_AVAILABLE = cv2 is not None


def preprocess_face(img):
    if cv2 is None:
        raise RuntimeError("OpenCV is not available in this environment.")
    resized = cv2.resize(img, (200, 200))
    return cv2.equalizeHist(resized)


def largest_face(gray_image):
    if cv2 is None:
        raise RuntimeError("OpenCV is not available in this environment.")
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.2, minNeighbors=5)
    if len(faces) == 0:
        return None
    return max(faces, key=lambda rect: rect[2] * rect[3])


def decode_image_bytes(raw_bytes):
    if cv2 is None:
        raise RuntimeError("OpenCV is not available in this environment.")
    image = cv2.imdecode(np.frombuffer(raw_bytes, np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Invalid image data")
    return image


def load_recognizer():
    if cv2 is None:
        raise RuntimeError("OpenCV is not available in this environment.")
    if not MODEL_FILE.exists() or not LABELS_FILE.exists():
        raise FileNotFoundError("Model not trained yet. Train the model first.")

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read(str(MODEL_FILE))
    with LABELS_FILE.open("r", encoding="utf-8") as f:
        labels = {int(k): v for k, v in json.load(f).items()}
    return recognizer, labels


def save_image_bytes(raw_bytes, file_path):
    image = Image.open(io.BytesIO(raw_bytes))
    image.convert("RGB").save(file_path, format="JPEG", quality=95)


def save_sample_images(name, mobile, uploads):
    folder_name = f"{sanitize_name(name)}_{sanitize_mobile(mobile)}"
    person_dir = DATA_DIR / folder_name
    person_dir.mkdir(parents=True, exist_ok=True)

    saved = 0
    for upload in uploads:
        raw = upload.getvalue()
        if not raw:
            continue
        file_path = person_dir / f"{folder_name}_{saved:03d}.jpg"
        if CV2_AVAILABLE:
            image = decode_image_bytes(raw)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            face = largest_face(gray)
            face_roi = (
                gray
                if face is None
                else gray[face[1] : face[1] + face[3], face[0] : face[0] + face[2]]
            )
            processed = preprocess_face(face_roi)
            cv2.imwrite(str(file_path), processed)
        else:
            save_image_bytes(raw, file_path)
        saved += 1

    return saved


def recognize_from_bytes(raw_bytes):
    if cv2 is None:
        raise RuntimeError("OpenCV is not available in this environment.")
    recognizer, labels = load_recognizer()
    image = decode_image_bytes(raw_bytes)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    face = largest_face(gray)
    if face is None:
        raise ValueError("No face found in image")

    x, y, w, h = face
    face_roi = gray[y : y + h, x : x + w]
    processed = preprocess_face(face_roi)
    label_id, confidence = recognizer.predict(processed)
    if confidence >= CONFIDENCE_THRESHOLD:
        return "unknown", confidence
    return labels.get(label_id, "unknown"), confidence


def ensure_state():
    defaults = {
        "samples": [],
        "attendance_image": None,
        "status": "Ready",
        "camera_mode": "environment",
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def add_sample(capture):
    if capture is None:
        st.warning("Capture a sample photo first.")
        return
    st.session_state.samples.append(capture)
    st.success(f"Sample added. Total samples: {len(st.session_state.samples)}")


def clear_samples():
    st.session_state.samples = []


def set_attendance_image(capture):
    if capture is None:
        st.warning("Capture a face photo first.")
        return
    st.session_state.attendance_image = capture
    st.success("Attendance photo ready.")


def clear_attendance():
    st.session_state.attendance_image = None


def render_location_capture(component_key):
    storage_key = f"attendance-location-{component_key}"
    payload = html(
        f"""
        <div style="border:1px solid rgba(128,128,128,.25);border-radius:12px;padding:12px;background:rgba(250,250,250,.02);">
          <div style="font-family:sans-serif;font-size:14px;margin-bottom:8px;">
            Capture current phone location for attendance
          </div>
          <button id="loc-btn" style="width:100%;padding:10px 12px;border:none;border-radius:10px;background:#0e7490;color:white;font-weight:600;cursor:pointer;">
            Capture current location
          </button>
          <div id="loc-status" style="font-family:sans-serif;font-size:13px;margin-top:8px;color:#475569;">
            Waiting for location permission.
          </div>
        </div>
        <script>
          const storageKey = {storage_key!r};
          const statusEl = document.getElementById("loc-status");
          const sendValue = (value) => {{
            window.parent.postMessage({{
              isStreamlitMessage: true,
              type: "streamlit:setComponentValue",
              value
            }}, "*");
          }};
          const setHeight = () => {{
            window.parent.postMessage({{
              isStreamlitMessage: true,
              type: "streamlit:setFrameHeight",
              height: document.body.scrollHeight + 16
            }}, "*");
          }};
          const setStatus = (message) => {{
            statusEl.textContent = message;
            setHeight();
          }};
          const cached = window.localStorage.getItem(storageKey);
          if (cached) {{
            try {{
              const parsed = JSON.parse(cached);
              setStatus("Using last captured location.");
              sendValue(parsed);
            }} catch (error) {{
              window.localStorage.removeItem(storageKey);
            }}
          }} else {{
            sendValue({{ status: "idle" }});
          }}
          document.getElementById("loc-btn").addEventListener("click", () => {{
            if (!navigator.geolocation) {{
              const value = {{ status: "unsupported", message: "Geolocation is not supported in this browser." }};
              window.localStorage.setItem(storageKey, JSON.stringify(value));
              setStatus(value.message);
              sendValue(value);
              return;
            }}
            setStatus("Requesting current location...");
            navigator.geolocation.getCurrentPosition(
              (position) => {{
                const coords = position.coords;
                const value = {{
                  status: "success",
                  lat: coords.latitude,
                  lon: coords.longitude,
                  accuracy: coords.accuracy,
                  captured_at: new Date().toISOString(),
                  message: "Location captured from this device."
                }};
                window.localStorage.setItem(storageKey, JSON.stringify(value));
                setStatus(`Location captured. Accuracy about ${{Math.round(coords.accuracy)}} meters.`);
                sendValue(value);
              }},
              (error) => {{
                const value = {{
                  status: "error",
                  message: error.message || "Unable to read location from this device."
                }};
                window.localStorage.setItem(storageKey, JSON.stringify(value));
                setStatus(value.message);
                sendValue(value);
              }},
              {{
                enableHighAccuracy: true,
                timeout: 15000,
                maximumAge: 0
              }}
            );
          }});
          window.addEventListener("load", setHeight);
          setHeight();
        </script>
        """,
        height=140,
    )
    return payload if isinstance(payload, dict) else {}


def render_stats():
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Employees", len(load_employees()))
    with col2:
        st.metric("Attendance rows", len(load_attendance_records()))
    with col3:
        st.metric("Status", st.session_state.status)


def register_employee(name, mobile, employee_id, role, company_name, logo_path, samples):
    if not name.strip():
        raise ValueError("Employee name is required.")
    if len(sanitize_mobile(mobile)) < 10:
        raise ValueError("Valid mobile number is required.")
    if not employee_id.strip():
        raise ValueError("Employee ID is required.")
    if not role.strip():
        raise ValueError("Role is required.")
    if not company_name.strip():
        raise ValueError("Company name is required.")
    if not samples:
        raise ValueError("Capture at least one sample photo first.")

    upsert_employee(name, sanitize_mobile(mobile), employee_id, role, company_name, logo_path)
    saved = save_sample_images(name, mobile, samples)
    append_capture_log(name, sanitize_mobile(mobile), f"{sanitize_name(name)}_{sanitize_mobile(mobile)}", saved)
    return saved


def render_employee_list():
    employees = load_employees()
    if not employees:
        st.info("No employees yet.")
        return
    for row in employees:
        st.write(f"**{row.get('Name', '')}**")
        st.caption(
            f"{row.get('EmployeeID', '')} | {row.get('Role', '')} | "
            f"{row.get('Mobile', '')} | {row.get('CompanyName', '')}"
        )
        st.divider()


def render_attendance_list():
    attendance = load_attendance_records()
    if not attendance:
        st.info("No attendance rows yet.")
        return
    for row in attendance:
        st.write(f"**{row.get('Name', '')}**")
        st.caption(
            f"{row.get('Date', '')} | IN {row.get('CheckIn', '')} | "
            f"OUT {row.get('CheckOut', '')} | Hours {row.get('WorkHours', '') or '-'}"
        )
        st.divider()


def attendance_location_fields(location_payload):
    if not isinstance(location_payload, dict):
        return "Unknown", "", ""
    if location_payload.get("status") != "success":
        return "Unknown", "", ""
    lat = location_payload.get("lat")
    lon = location_payload.get("lon")
    accuracy = location_payload.get("accuracy")
    label = "Browser GPS"
    if accuracy not in (None, ""):
        label = f"{label} (+/- {float(accuracy):.0f} m)"
    lat_value = f"{float(lat):.6f}" if lat not in (None, "") else ""
    lon_value = f"{float(lon):.6f}" if lon not in (None, "") else ""
    return label, lat_value, lon_value


st.set_page_config(page_title=APP_TITLE, page_icon="camera", layout="wide")
ensure_state()

st.title("Attendance Web App")
st.caption("Open this link on mobile or desktop. No install needed.")
if CV2_AVAILABLE:
    st.info(
        "For browser camera access on a public link, the site should run over HTTPS. "
        "On localhost, camera access works without HTTPS."
    )
else:
    st.warning(
        "OpenCV is not available on this host, so the app is running in simple mode. "
        "You can still register employees and mark attendance manually from the browser link."
    )

with st.sidebar:
    st.header("Server")
    st.write(f"Root: `{ROOT_DIR}`")
    st.write("Use the same link for everyone once deployed.")
    if st.button("Refresh dashboard", use_container_width=True):
        st.rerun()
    st.divider()
    st.write("Camera mode")
    st.session_state.camera_mode = st.selectbox(
        "Facing mode",
        options=["environment", "user"],
        index=0 if st.session_state.camera_mode == "environment" else 1,
        label_visibility="collapsed",
    )
    if not CV2_AVAILABLE:
        st.caption("Camera mode is kept for future compatibility, but recognition is manual in this deployment.")

render_stats()

col_backend, col_camera = st.columns([1, 1])

with col_backend:
    st.subheader("Train and data")
    if st.button(
        "Train Model",
        type="primary",
        use_container_width=True,
        disabled=not CV2_AVAILABLE,
        help="OpenCV is required for model training on this app host.",
    ):
        try:
            train_faces()
            st.session_state.status = "Model trained"
            st.success("Model trained successfully.")
        except Exception as exc:
            st.session_state.status = "Train failed"
            st.error(str(exc))
    if not CV2_AVAILABLE:
        st.caption("Training is disabled in Streamlit Cloud simple mode.")

    st.subheader("Register employee")
    name = st.text_input("Name", key="employee_name")
    mobile = st.text_input("Mobile", key="employee_mobile")
    employee_id = st.text_input("Employee ID", key="employee_id")
    role = st.selectbox("Role", ROLES, key="employee_role")
    company_name = st.text_input("Company name", value=DEFAULT_COMPANY_NAME, key="company_name")
    logo_path = st.text_input("Logo path optional", value="", key="logo_path")
    sample_photo = st.camera_input("Capture sample photo", key="sample_camera")
    extra_uploads = st.file_uploader(
        "Or add more sample photos",
        type=["png", "jpg", "jpeg"],
        accept_multiple_files=True,
        key="sample_uploads",
    )

    if sample_photo and st.button("Add camera sample", use_container_width=True, key="add_camera_sample"):
        add_sample(sample_photo)

    if extra_uploads and st.button("Add uploaded samples", use_container_width=True, key="add_uploaded_samples"):
        st.session_state.samples.extend(extra_uploads)
        st.success(f"Added {len(extra_uploads)} uploaded sample(s).")

    if st.session_state.samples:
        st.write(f"Captured samples: {len(st.session_state.samples)}")
        st.image([sample.getvalue() for sample in st.session_state.samples], width=120)
        if st.button("Clear samples", use_container_width=True, key="clear_samples"):
            clear_samples()
            st.rerun()

    if st.button("Register employee", type="primary", use_container_width=True, key="register_employee"):
        try:
            saved = register_employee(
                name,
                mobile,
                employee_id,
                role,
                company_name,
                logo_path,
                st.session_state.samples,
            )
            st.session_state.status = "Employee registered"
            st.success(f"Registered employee and saved {saved} sample(s).")
            clear_samples()
        except Exception as exc:
            st.session_state.status = "Register failed"
            st.error(str(exc))

with col_camera:
    st.subheader("Mark attendance")
    location_payload = render_location_capture("attendance")
    if isinstance(location_payload, dict):
        if location_payload.get("status") == "success":
            st.caption(
                "Location ready: "
                f"{float(location_payload.get('lat')):.6f}, "
                f"{float(location_payload.get('lon')):.6f}"
            )
        elif location_payload.get("status") == "error":
            st.caption(f"Location not captured: {location_payload.get('message', 'Permission denied.')}")
        elif location_payload.get("status") == "unsupported":
            st.caption("This browser does not support GPS capture.")

    if CV2_AVAILABLE:
        attendance_photo = st.camera_input("Capture attendance photo", key="attendance_camera")
        uploaded_attendance = st.file_uploader(
            "Or upload attendance photo",
            type=["png", "jpg", "jpeg"],
            accept_multiple_files=False,
            key="attendance_upload",
        )

        if attendance_photo and st.button("Use camera photo", use_container_width=True, key="use_camera_photo"):
            set_attendance_image(attendance_photo)

        if uploaded_attendance and st.button("Use uploaded photo", use_container_width=True, key="use_uploaded_photo"):
            set_attendance_image(uploaded_attendance)

        if st.session_state.attendance_image is not None:
            st.image(st.session_state.attendance_image, width=220)
            if st.button("Mark Attendance", type="primary", use_container_width=True, key="mark_attendance"):
                try:
                    raw = st.session_state.attendance_image.getvalue()
                    name, confidence = recognize_from_bytes(raw)
                    if name == "unknown":
                        st.warning(f"Face not recognized. Confidence: {confidence:.1f}")
                    else:
                        today_file = ATTENDANCE_DIR / f"attendance_{datetime.now().strftime('%Y%m%d')}.csv"
                        today_file.parent.mkdir(parents=True, exist_ok=True)
                        if not today_file.exists():
                            with today_file.open("w", newline="", encoding="utf-8") as f:
                                writer = csv.writer(f)
                                writer.writerow(ATTENDANCE_COLUMNS)
                        location_name, lat, lon = attendance_location_fields(location_payload)
                        message, marked = mark_attendance(name, today_file, location_name, lat, lon)
                        st.session_state.status = "Attendance marked" if marked else "Attendance blocked"
                        st.success(message)
                        clear_attendance()
                except Exception as exc:
                    st.session_state.status = "Attendance failed"
                    st.error(str(exc))
    else:
        employees = load_employees()
        employee_names = sorted({row.get("Name", "").strip() for row in employees if row.get("Name", "").strip()})
        if employee_names:
            selected_name = st.selectbox("Select employee", employee_names, key="manual_attendance_name")
            if st.button("Mark Attendance Manually", type="primary", use_container_width=True, key="manual_mark_attendance"):
                try:
                    today_file = ATTENDANCE_DIR / f"attendance_{datetime.now().strftime('%Y%m%d')}.csv"
                    today_file.parent.mkdir(parents=True, exist_ok=True)
                    if not today_file.exists():
                        with today_file.open("w", newline="", encoding="utf-8") as f:
                            writer = csv.writer(f)
                            writer.writerow(ATTENDANCE_COLUMNS)
                    message, marked = mark_attendance(selected_name, today_file)
                    st.session_state.status = "Attendance marked" if marked else "Attendance blocked"
                    st.success(message)
                except Exception as exc:
                    st.session_state.status = "Attendance failed"
                    st.error(str(exc))
        else:
            st.info("Add employees first to use manual attendance mode.")

st.divider()
left, right = st.columns(2)
with left:
    st.subheader("Employees")
    render_employee_list()
with right:
    st.subheader("Attendance")
    render_attendance_list()
