# Face Recognition Attendance System

This project now works as a browser-based app so everyone can use it from mobile or desktop with one link.

## Quick Start

This repo already contains a working Python 3.11 build cache in `.build311` and `.build_lib`.
Use that environment instead of the broken `.venv`:

```powershell
cd "C:\attendence system"
.\start_app.ps1
```

Open it in a browser:

- Same PC: `http://127.0.0.1:8000/`
- Phone on same Wi-Fi: `http://<your-pc-lan-ip>:8000/`

## How to use

1. Open the browser link.
2. Enter the employee details.
3. Capture a face sample or use the phone camera fallback.
4. Register the employee.
5. Train the model.
6. Capture a face photo and mark attendance.

## Project Folders

See [PROJECT_STRUCTURE.md](./PROJECT_STRUCTURE.md) for the folder layout.

## Browser App

The main app lives in [server/app.py](./server/app.py). It serves:
- the web page
- the JSON API
- the camera upload routes

## Streamlit App

If you want a Streamlit version for deployment, run:

```powershell
cd "C:\attendence system"
.\start_streamlit.ps1
```

The Streamlit app is in [streamlit_app.py](./streamlit_app.py).
For a public hosted app, Streamlit is often the simplest path because it gives you a shareable link and a built-in browser UI.
If the host does not provide OpenCV, the app still starts in simple mode so the link works, and you can mark attendance manually from the browser.

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

## Android app

The Android client still lives in `android-app/`, but the browser version is the quickest no-install option.

## Notes

- Press `q` in the camera window to stop.
- If recognition is weak, increase image samples and retrain.
- You can tune `CONFIDENCE_THRESHOLD` in `src/mark_attendance.py` (lower is stricter).
