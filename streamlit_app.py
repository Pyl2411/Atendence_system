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
ADMIN_ROLES = {"Manager", "Co Founder"}
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
        "attendance_mode": False,
        "log_lines": [],
        "viewer_role": ROLES[0],
        "viewer_query": "",
        "sample_target": 5,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def log(message):
    st.session_state.log_lines.append(message)


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


def get_visible_data(role, query, employees, attendance):
    visible_employees = employees
    visible_attendance = attendance
    if role not in ADMIN_ROLES:
        if query:
            visible_employees = [row for row in employees if _matches_query(row, query)]
            visible_attendance = [row for row in attendance if _matches_query(row, query)]
        else:
            visible_employees = []
            visible_attendance = []

    access_scope = "Full access" if role in ADMIN_ROLES else "Filtered access"
    if role in ADMIN_ROLES:
        hint = "Manager and Co Founder can see every employee row and every attendance entry."
    elif query:
        hint = "Filtered personal view. Search by name, mobile, or employee ID to show your record."
    else:
        hint = "Enter your name, mobile, or employee ID to see your record."

    return visible_employees, visible_attendance, access_scope, hint


def render_data_frame(rows, columns):
    if not rows:
        st.info("No rows to show.")
        return

    table = [{column: row.get(column, "") for column in columns} for row in rows]
    st.dataframe(table, use_container_width=True)


def render_dashboard_tab():
    st.subheader("Dashboard")
    st.write("Role-aware viewing for employees and attendance records.")

    col1, col2, col3 = st.columns([2, 3, 1])
    with col1:
        st.selectbox("Viewer role", ROLES, key="viewer_role")
    with col2:
        st.text_input("Search employee", key="viewer_query")
    with col3:
        if st.button("Refresh", use_container_width=True, key="dashboard_refresh"):
            st.experimental_rerun()

    employees = load_employees()
    attendance = load_attendance_records()
    visible_employees, visible_attendance, access_scope, hint = get_visible_data(
        st.session_state.viewer_role,
        st.session_state.viewer_query,
        employees,
        attendance,
    )

    stats_col1, stats_col2, stats_col3 = st.columns(3)
    with stats_col1:
        st.metric("Employees in view", len(visible_employees))
    with stats_col2:
        st.metric("Attendance rows", len(visible_attendance))
    with stats_col3:
        st.metric("Total employees", len(employees))

    st.info(f"Current access: {access_scope}. {hint}")

    st.markdown("#### Employees")
    render_data_frame(visible_employees, ["Name", "Mobile", "EmployeeID", "Role", "CompanyName"])

    st.markdown("#### Attendance")
    render_data_frame(visible_attendance, ["Date", "Name", "CheckIn", "CheckOut", "WorkHours", "SourceFile"])


def render_activity_tab():
    st.subheader("System Activity")
    if st.button("Clear activity log", use_container_width=True, key="clear_activity"):
        st.session_state.log_lines = []

    if st.session_state.log_lines:
        st.text_area(
            "Activity log",
            value="\n".join(st.session_state.log_lines),
            height=420,
            disabled=True,
        )
    else:
        st.info(
            "Ready. Use Register to enroll people, Dashboard to review data, and Activity to watch command output."
        )


def render_register_tab():
    st.subheader("Employee Registration")

    left, right = st.columns([2, 1])
    with left:
        name = st.text_input("Employee Name", key="employee_name")
        mobile = st.text_input("Mobile Number", key="employee_mobile")
        employee_id = st.text_input("Employee ID", key="employee_id")
        role = st.selectbox("Role", ROLES, index=ROLES.index(st.session_state.get("employee_role", ROLES[2])), key="employee_role")
        company_name = st.text_input("Company Name", value=DEFAULT_COMPANY_NAME, key="company_name")
        logo_path = st.text_input("Logo Path", key="logo_path")
        sample_target = st.number_input(
            "Face Samples",
            min_value=1,
            max_value=20,
            value=st.session_state.sample_target,
            key="sample_target",
        )

        button_row = st.columns([1, 1, 1])
        if button_row[0].button("Register & Capture", type="primary", use_container_width=True, key="register_capture"):
            try:
                if len(st.session_state.samples) < sample_target:
                    raise ValueError(
                        f"Please capture at least {sample_target} sample photo(s) before registering."
                    )
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
                log(f"Registered employee: {name} ({mobile})")
                clear_samples()
            except Exception as exc:
                st.session_state.status = "Register failed"
                st.error(str(exc))
                log(f"Register failed: {exc}")

        if button_row[1].button("Train Model", use_container_width=True, key="register_train"):
            try:
                with st.spinner("Training model..."):
                    train_faces()
                st.session_state.status = "Model trained"
                st.success("Model trained successfully.")
                log("Model trained successfully.")
            except Exception as exc:
                st.session_state.status = "Train failed"
                st.error(str(exc))
                log(f"Training failed: {exc}")

        if button_row[2].button("Start Attendance", use_container_width=True, key="register_attendance"):
            st.session_state.attendance_mode = True
            st.experimental_rerun()

        st.markdown(
            """
            1. Register an employee once with name, mobile, employee ID, and role.
            2. Capture face samples from the webcam or upload sample photos.
            3. Train the model.
            4. Start attendance to mark IN/OUT.

            Manager and Co Founder can view every employee and every attendance record from the Dashboard tab. Other roles can filter to their own record.
            """
        )

    with right:
        st.subheader("Face samples")
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
                st.experimental_rerun()

        st.metric("Current access", "Full access" if st.session_state.viewer_role in ADMIN_ROLES else "Filtered access")
        st.caption("Tip: leave Logo Path empty if you do not have a company logo file.")

    if st.session_state.attendance_mode:
        st.divider()
        st.subheader("Attendance")
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
                            log(f"Attendance: {message}")
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
                        log(f"Attendance: {message}")
                    except Exception as exc:
                        st.session_state.status = "Attendance failed"
                        st.error(str(exc))
            else:
                st.info("Add employees first to use manual attendance mode.")
        if st.button("Close Attendance", use_container_width=True, key="close_attendance"):
            st.session_state.attendance_mode = False
            st.experimental_rerun()


def main():
    tab_register, tab_dashboard, tab_activity = st.tabs(["Register", "Dashboard", "Activity"])

    with tab_register:
        render_register_tab()
    with tab_dashboard:
        render_dashboard_tab()
    with tab_activity:
        render_activity_tab()


st.set_page_config(page_title=APP_TITLE, page_icon="camera", layout="wide")
ensure_state()

st.title("Vickhardth Automation - Employee Access Portal")
st.caption("FACE RECOGNITION ATTENDANCE WITH ROLE-AWARE EMPLOYEE ACCESS")
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
        st.experimental_rerun()
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

main()
