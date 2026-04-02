import csv
from datetime import datetime
from pathlib import Path

import streamlit as st

from src.data_store import ATTENDANCE_DIR, DATA_DIR, ROOT_DIR, load_attendance_records, load_employees


APP_TITLE = "Attendance Web App"
DEFAULT_COMPANY_NAME = "Vickhardth Automation"
ROLES = ["Manager", "Co Founder", "Employee", "Team Leader", "Trainee"]
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


def _clean(value):
    return (value or "").strip()


def ensure_state():
    if "status" not in st.session_state:
        st.session_state.status = "Ready"


def employees_file():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    return DATA_DIR / "employees.csv"


def save_employee(name, mobile, employee_id, role, company_name):
    file_path = employees_file()
    rows = []
    if file_path.exists():
        with file_path.open("r", newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))

    mobile = "".join(ch for ch in mobile if ch.isdigit())
    if not name.strip():
        raise ValueError("Employee name is required.")
    if len(mobile) < 10:
        raise ValueError("Valid mobile number is required.")
    if not employee_id.strip():
        raise ValueError("Employee ID is required.")
    if not role.strip():
        raise ValueError("Role is required.")
    if not company_name.strip():
        raise ValueError("Company name is required.")

    for row in rows:
        if _clean(row.get("Mobile")).strip() == mobile:
            raise ValueError(f"Duplicate mobile number not allowed: {mobile}")
        if _clean(row.get("EmployeeID")).lower() == employee_id.strip().lower():
            raise ValueError(f"Duplicate employee ID not allowed: {employee_id}")

    rows.append(
        {
            "Name": name.strip(),
            "Mobile": mobile,
            "EmployeeID": employee_id.strip(),
            "Role": role.strip(),
            "CompanyName": company_name.strip(),
            "LogoPath": "",
        }
    )

    with file_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["Name", "Mobile", "EmployeeID", "Role", "CompanyName", "LogoPath"],
        )
        writer.writeheader()
        writer.writerows(rows)


def ensure_attendance_file(file_path):
    file_path.parent.mkdir(parents=True, exist_ok=True)
    if file_path.exists():
        return
    with file_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=ATTENDANCE_COLUMNS)
        writer.writeheader()


def read_rows(file_path):
    rows = []
    if not file_path.exists():
        return rows
    with file_path.open("r", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows.append(
                {
                    "Name": _clean(row.get("Name")),
                    "Date": _clean(row.get("Date")),
                    "CheckIn": _clean(row.get("CheckIn") or row.get("Time")),
                    "CheckInLocation": _clean(row.get("CheckInLocation")),
                    "CheckInLat": _clean(row.get("CheckInLat")),
                    "CheckInLon": _clean(row.get("CheckInLon")),
                    "CheckOut": _clean(row.get("CheckOut")),
                    "CheckOutLocation": _clean(row.get("CheckOutLocation")),
                    "CheckOutLat": _clean(row.get("CheckOutLat")),
                    "CheckOutLon": _clean(row.get("CheckOutLon")),
                    "WorkHours": _clean(row.get("WorkHours")),
                }
            )
    return rows


def write_rows(file_path, rows):
    with file_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=ATTENDANCE_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def mark_attendance_manual(name):
    today_file = ATTENDANCE_DIR / f"attendance_{datetime.now().strftime('%Y%m%d')}.csv"
    ensure_attendance_file(today_file)
    rows = read_rows(today_file)
    now = datetime.now()
    today = now.strftime("%Y-%m-%d")
    now_time = now.strftime("%H:%M:%S")

    for row in rows:
        if row.get("Name") != name or row.get("Date") != today:
            continue
        if row.get("CheckIn") and row.get("CheckOut"):
            return f"{name}: already marked IN and OUT today.", True
        if row.get("CheckIn") and not row.get("CheckOut"):
            row["CheckOut"] = now_time
            row["WorkHours"] = "manual"
            write_rows(today_file, rows)
            return f"{name}: OUT at {now_time}", True

    rows.append(
        {
            "Name": name,
            "Date": today,
            "CheckIn": now_time,
            "CheckInLocation": "",
            "CheckInLat": "",
            "CheckInLon": "",
            "CheckOut": "",
            "CheckOutLocation": "",
            "CheckOutLat": "",
            "CheckOutLon": "",
            "WorkHours": "",
        }
    )
    write_rows(today_file, rows)
    return f"{name}: IN at {now_time}", True


def render_stats():
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Employees", len(load_employees()))
    with col2:
        st.metric("Attendance rows", len(load_attendance_records()))
    with col3:
        st.metric("Status", st.session_state.status)


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
            f"OUT {row.get('CheckOut', '')} | Source {row.get('SourceFile', '')}"
        )
        st.divider()


st.set_page_config(page_title=APP_TITLE, page_icon="camera", layout="wide")
ensure_state()

st.title("Attendance Web App")
st.caption("One browser link for everyone. No install needed.")
st.info(
    "This cloud version is a simple attendance dashboard. "
    "It is designed to be reliable on Streamlit Cloud and work from mobile or desktop."
)

with st.sidebar:
    st.header("Cloud App")
    st.write(f"Root: `{ROOT_DIR}`")
    st.write("Works from the same public link for everyone.")
    if st.button("Refresh dashboard", use_container_width=True):
        st.rerun()

render_stats()

col_left, col_right = st.columns(2)

with col_left:
    st.subheader("Register employee")
    with st.form("register_form", clear_on_submit=True):
        name = st.text_input("Name")
        mobile = st.text_input("Mobile")
        employee_id = st.text_input("Employee ID")
        role = st.selectbox("Role", ROLES)
        company_name = st.text_input("Company name", value=DEFAULT_COMPANY_NAME)
        submitted = st.form_submit_button("Register employee", use_container_width=True)
        if submitted:
            try:
                save_employee(name, mobile, employee_id, role, company_name)
                st.session_state.status = "Employee registered"
                st.success("Employee registered successfully.")
                st.rerun()
            except Exception as exc:
                st.session_state.status = "Register failed"
                st.error(str(exc))

    st.subheader("Mark attendance")
    employees = load_employees()
    employee_names = sorted({row.get("Name", "").strip() for row in employees if row.get("Name", "").strip()})
    if employee_names:
        selected_name = st.selectbox("Select employee", employee_names)
        if st.button("Mark attendance", type="primary", use_container_width=True):
            try:
                message, _ = mark_attendance_manual(selected_name)
                st.session_state.status = "Attendance marked"
                st.success(message)
                st.rerun()
            except Exception as exc:
                st.session_state.status = "Attendance failed"
                st.error(str(exc))
    else:
        st.info("Register employees first.")

with col_right:
    st.subheader("Employees")
    render_employee_list()
    st.subheader("Attendance")
    render_attendance_list()
