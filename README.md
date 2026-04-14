# Daily Attendance System - Face Recognition Web App

A modern web-based attendance management system using face recognition technology. The app opens the camera immediately upon loading, functioning like a biometric attendance machine.

## Features

- 📷 **Immediate Camera Access**: Camera opens automatically when the app loads
- 👥 **Role-Based Access**: Different permissions for Managers, Employees, etc.
- 📊 **Daily Presence Tracking**: Real-time attendance statistics
- 🔒 **Secure Face Recognition**: OpenCV-based facial recognition
- 📱 **Mobile Friendly**: Works on phones and desktops
- 🗺️ **Location Tracking**: Optional GPS location capture

## Quick Start

### Prerequisites
- Python 3.8+
- Webcam/Camera access

### Installation

1. Clone or download the project
2. Install dependencies:
```bash
pip install -r requirements.txt
```

### Running the App

#### For Development
```powershell
.\start_streamlit.ps1
```

#### For Production Deployment
```bash
streamlit run streamlit_app.py --server.port 8501 --server.address 0.0.0.0
```

Open `http://localhost:8501` in your browser.

## How to Use

1. **Select Your Role**: Choose your role from the dropdown (Manager/Co Founder see all records)
2. **Camera Access**: The camera will open automatically
3. **Mark Attendance**: Position your face and click "Mark Attendance"
4. **View Records**: See your attendance history based on your role permissions

## User Roles

- **Manager/Co Founder**: Full access to all employee records and statistics
- **Employee/Team Leader/Trainee**: Access to personal attendance records only

## Project Structure

- `streamlit_app.py` - Main web application
- `src/` - Core functionality modules
- `data/` - Employee data and face samples
- `models/` - Trained face recognition models
- `attendance/` - Daily attendance CSV files

## Deployment

The app is ready for deployment on:
- Streamlit Cloud
- Heroku
- AWS/GCP/Azure
- Local servers

For production deployment, ensure:
- HTTPS enabled for camera access
- Sufficient storage for face data
- Regular backups of attendance records

## Security Notes

- Face recognition requires camera permissions
- Location data is optional and stored locally
- All data is stored in CSV files (consider database for production)
- No external APIs required for basic functionality

## Troubleshooting

- **Camera not working**: Ensure HTTPS in production, check browser permissions
- **Face not recognized**: Retrain the model with better lighting/samples
- **No records showing**: Check role permissions and data files

---

© 2024 Vickhardth Automation - Secure Attendance System
For a public hosted app, Streamlit is often the simplest path because it gives you a shareable link and a built-in browser UI.
If the host does not provide OpenCV, the app still starts in simple mode so the link works, and you can mark attendance manually from the browser.
Streamlit Cloud should use the cloud-safe [requirements.txt](./requirements.txt). If you want the OpenCV desktop path on your PC, use [requirements-desktop.txt](./requirements-desktop.txt).
On mobile, users should press the location button before marking attendance so the app stores their phone GPS coordinates instead of only the server location.

## Desktop GUI

The desktop GUI still exists for local machine use, but the browser app is the easiest way for everyone to use it.

## Optional legacy scripts

You can still run the old Python scripts directly if needed.

## Capture face samples

Capture images for each person:

```powershell
python src/capture_faces.py --name om --samples 40
python src/capture_faces.py --name rahul --samples 40
```

Captured images are saved in `data/<person_name>/`.

## 3) Train recognizer

```powershell
python src/train_model.py
```

This creates:
- `models/face_trainer.yml`
- `models/labels.json`

## 4) Start attendance

```powershell
python src/mark_attendance.py
```

Attendance is saved in `attendance/attendance_YYYYMMDD.csv`.

## Optional: GUI app

Use the desktop app for employee registration, face capture, model training, attendance, and role-aware data review:

```powershell
python src/gui_app.py
```

In the GUI:
- Use the `Register` tab to enroll an employee.
- Use the `Dashboard` tab to search employees and review attendance.
- Managers and cofounders can see every employee record and every attendance row.
- Use the `Activity` tab to watch command output.

Notes:
- The `Logo Path` field is optional. Leave it blank if you do not have a logo file.
- The scripts now resolve project paths from the app directory, so they work even when launched from the packaged app.

## Shared backend for browser/mobile

Run the HTTP API and browser app with the same command:

```powershell
.\start_app.ps1
```

API endpoints:
- `GET /health`
- `GET /employees`
- `GET /attendance`
- `POST /employees/register`
- `POST /train`
- `POST /attendance/mark`

If the live camera does not start on a phone, use the capture buttons. They fall back to the phone camera picker.

## Deployment

See [DEPLOYMENT.md](./DEPLOYMENT.md) for local, LAN, and public deployment steps.

For browser camera access on a public link, deploy with HTTPS or behind an HTTPS reverse proxy.

For Streamlit deployment, use HTTPS as well if you want camera access to work reliably on phones.
If Streamlit Cloud cannot use OpenCV on its default Python version, the app falls back to a simpler browser mode instead of crashing.
Browser GPS capture also needs the user to allow location permission in the phone browser.

## Android app

The Android client still lives in `android-app/`, but the browser version is the quickest no-install option.

## Notes

- Press `q` in the camera window to stop.
- If recognition is weak, increase image samples and retrain.
- You can tune `CONFIDENCE_THRESHOLD` in `src/mark_attendance.py` (lower is stricter).
