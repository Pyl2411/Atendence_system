import io
import csv
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import streamlit as st

try:
    import cv2
except ImportError:  # pragma: no cover - optional in Streamlit Cloud
    cv2 = None

from src.capture_faces import sanitize_mobile, sanitize_name
from src.data_store import ATTENDANCE_DIR, load_attendance_records, load_employees
from src.mark_attendance import ATTENDANCE_COLUMNS, CONFIDENCE_THRESHOLD, mark_attendance
from src.train_model import LABELS_FILE, MODEL_FILE


APP_TITLE = 'Daily Attendance System'
DEFAULT_COMPANY_NAME = 'Vickhardth Automation'
ROLES = ['Manager', 'Co Founder', 'Employee', 'Team Leader', 'Trainee']
ADMIN_ROLES = {'Manager', 'Co Founder'}
CV2_AVAILABLE = cv2 is not None


def preprocess_face(img):
    if cv2 is None:
        raise RuntimeError('OpenCV is not available in this environment.')
    resized = cv2.resize(img, (200, 200))
    return cv2.equalizeHist(resized)


def largest_face(gray_image):
    if cv2 is None:
        raise RuntimeError('OpenCV is not available in this environment.')
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    )
    faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.2, minNeighbors=5)
    if len(faces) == 0:
        return None
    return max(faces, key=lambda rect: rect[2] * rect[3])


def decode_image_bytes(raw_bytes):
    if cv2 is None:
        raise RuntimeError('OpenCV is not available in this environment.')
    image = cv2.imdecode(np.frombuffer(raw_bytes, np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError('Invalid image data')
    return image


def load_recognizer():
    if cv2 is None:
        raise RuntimeError('OpenCV is not available in this environment.')
    if not MODEL_FILE.exists() or not LABELS_FILE.exists():
        raise FileNotFoundError('Model not trained yet. Train the model first.')

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read(str(MODEL_FILE))
    with LABELS_FILE.open('r', encoding='utf-8') as f:
        labels = {int(k): v for k, v in json.load(f).items()}
    return recognizer, labels


def recognize_from_bytes(raw_bytes):
    if cv2 is None:
        raise RuntimeError('OpenCV is not available in this environment.')
    recognizer, labels = load_recognizer()
    image = decode_image_bytes(raw_bytes)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    face = largest_face(gray)
    if face is None:
        raise ValueError('No face found in image')

    x, y, w, h = face
    face_roi = gray[y : y + h, x : x + w]
    processed = preprocess_face(face_roi)
    label_id, confidence = recognizer.predict(processed)
    if confidence >= CONFIDENCE_THRESHOLD:
        return 'unknown', confidence
    return labels.get(label_id, 'unknown'), confidence


def ensure_state():
    defaults = {
        'attendance_image': None,
        'camera_mode': 'environment',
        'user_role': ROLES[2],  # Default to Employee
        'user_name': None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def get_user_role_info():
    role = st.session_state.user_role
    is_admin = role in ADMIN_ROLES
    return role, is_admin


def calculate_daily_presence(attendance_records, role):
    today = datetime.now().strftime('%Y-%m-%d')
    today_records = [r for r in attendance_records if r.get('Date') == today]

    if role in ADMIN_ROLES:
        all_names = set(r.get('Name', '') for r in attendance_records if r.get('Name', ''))
        total_employees = len(all_names)
        present_today = len(set(r.get('Name', '') for r in today_records if r.get('CheckIn', '')))
        checked_out = len([r for r in today_records if r.get('CheckOut', '')])
    else:
        user_name = st.session_state.get('user_name')
        if user_name:
            user_records = [r for r in attendance_records if r.get('Name') == user_name]
            total_employees = 1
            present_today = 1 if any(r.get('CheckIn', '') for r in user_records if r.get('Date') == today) else 0
            checked_out = len([r for r in user_records if r.get('Date') == today and r.get('CheckOut', '')])
        else:
            total_employees = 0
            present_today = 0
            checked_out = 0

    return {
        'total_employees': total_employees,
        'present_today': present_today,
        'checked_out': checked_out,
        'absent_today': max(0, total_employees - present_today)
    }


def render_attendance_camera():
    st.subheader('📷 Face Recognition Attendance')

    col1, col2 = st.columns([3, 1])
    with col1:
        st.selectbox('Your Role', ROLES, key='user_role')
    with col2:
        if st.button('🔄 Refresh', use_container_width=True):
            st.experimental_rerun()

    role, is_admin = get_user_role_info()

    employees = load_employees()
    attendance = load_attendance_records()
    stats = calculate_daily_presence(attendance, role)

    st.markdown('### 📊 Daily Attendance Overview')
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric('Total Employees', stats['total_employees'])
    with col2:
        st.metric('Present Today', stats['present_today'])
    with col3:
        st.metric('Checked Out', stats['checked_out'])
    with col4:
        st.metric('Absent Today', stats['absent_today'])

    st.markdown('---')

    if CV2_AVAILABLE:
        st.markdown('### 🎯 Mark Your Attendance')

        attendance_photo = st.camera_input(
            'Position your face in the camera and click Mark Attendance',
            key='attendance_camera'
        )

        if attendance_photo:
            st.session_state.attendance_image = attendance_photo
            st.image(attendance_photo, width=300, caption='Captured Image')

            if st.button('✅ Mark Attendance', type='primary', use_container_width=True):
                try:
                    raw = attendance_photo.getvalue()
                    name, confidence = recognize_from_bytes(raw)

                    if name == 'unknown':
                        st.error('❌ Face not recognized. Please try again or contact administrator.')
                    else:
                        st.session_state.user_name = name

                        today_file = ATTENDANCE_DIR / f'attendance_{datetime.now().strftime("%Y%m%d")}.csv'
                        today_file.parent.mkdir(parents=True, exist_ok=True)
                        if not today_file.exists():
                            with today_file.open('w', newline='', encoding='utf-8') as f:
                                writer = csv.writer(f)
                                writer.writerow(ATTENDANCE_COLUMNS)

                        message, marked = mark_attendance(name, today_file)
                        if marked:
                            st.success(f'✅ {message}')
                            st.balloons()
                        else:
                            st.warning(f'⚠️ {message}')

                        st.session_state.attendance_image = None
                        st.experimental_rerun()

                except Exception as exc:
                    st.error(f'❌ Error processing attendance: {str(exc)}')
    else:
        st.warning('⚠️ Camera not available. Please use a device with camera support.')


def render_attendance_records():
    st.subheader('📋 Attendance Records')

    role, is_admin = get_user_role_info()
    employees = load_employees()
    attendance = load_attendance_records()

    if not is_admin and not st.session_state.get('user_name'):
        st.info('👤 Please mark your attendance first to view your records.')
        return

    if is_admin:
        visible_records = attendance
        st.info('👑 Showing all employee records (Admin access)')
    else:
        user_name = st.session_state.get('user_name')
        visible_records = [r for r in attendance if r.get('Name') == user_name]
        st.info(f'👤 Showing your records only')

    if not visible_records:
        st.info('📭 No attendance records found.')
        return

    records_data = []
    for record in visible_records[-20:]:
        records_data.append({
            'Date': record.get('Date', ''),
            'Name': record.get('Name', ''),
            'Check In': record.get('CheckIn', ''),
            'Check Out': record.get('CheckOut', ''),
            'Work Hours': record.get('WorkHours', ''),
            'Location': record.get('CheckInLocation', '')
        })

    st.dataframe(records_data, use_container_width=True)


def main():
    st.set_page_config(
        page_title=APP_TITLE,
        page_icon='📷',
        layout='wide',
        initial_sidebar_state='collapsed'
    )

    ensure_state()

    st.title('🏢 Daily Attendance System')
    st.caption('Face Recognition Based Attendance Management')

    render_attendance_camera()

    st.markdown('---')

    render_attendance_records()

    st.markdown('---')
    st.caption('© 2024 Vickhardth Automation - Secure Attendance System')


if __name__ == '__main__':
    main()
