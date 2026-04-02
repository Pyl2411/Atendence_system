# Deployment Guide

This project is now a browser-based app. That means users do not install anything on their phones.

## Local use

Run on your PC:

```powershell
cd "C:\attendence system"
.\start_app.ps1
```

Open in a browser:

- Same PC: `http://127.0.0.1:8000/`
- Phone on same Wi-Fi: `http://<your-pc-lan-ip>:8000/`

## LAN use

If your phone is on the same Wi-Fi:

1. Start the server with `start_app.ps1`
2. Find your PC IP address with `ipconfig`
3. Open `http://<your-pc-lan-ip>:8000/` on the phone

If Windows Firewall blocks the port, allow Python on private networks or open port `8000`.

## Public deployment

To make it usable from anywhere:

1. Put the project on a Windows or Linux machine with a public IP.
2. Run the server process continuously.
3. Expose port `8000` through a firewall or reverse proxy.
4. Point users to `http://your-domain-or-ip:8000/`.

Recommended production shape:

- Run behind a reverse proxy like Nginx or IIS.
- Keep the `data/`, `attendance/`, and `models/` folders on persistent storage.
- Back up CSV/model files regularly.

## HTTPS for camera support

Public browser camera access needs HTTPS.

Two good options:

1. Terminate HTTPS in a reverse proxy
- Nginx, IIS, Caddy, Cloudflare Tunnel, or a cloud load balancer
- The proxy serves `https://your-domain/`
- It forwards traffic to the Python app on `http://127.0.0.1:8000`

2. Run the Python app with a certificate
- Set `ATTENDANCE_CERT_FILE` and `ATTENDANCE_KEY_FILE`
- Start the app normally
- The app serves HTTPS directly on port `8000`

Example environment variables:

```powershell
$env:ATTENDANCE_CERT_FILE="C:\certs\fullchain.pem"
$env:ATTENDANCE_KEY_FILE="C:\certs\privkey.pem"
.\start_app.ps1
```

## Streamlit deployment

You can also deploy a Streamlit version if that is easier for you.

Local run:

```powershell
cd "C:\attendence system"
.\start_streamlit.ps1
```

For public deployment:

Option A: Streamlit Community Cloud

1. Push this repo to GitHub.
2. Go to Streamlit Community Cloud and create a new app.
3. Set the main file to `app.py`.
4. Make sure Streamlit uses the cloud-safe `requirements.txt` in the repo root.
5. Deploy and share the HTTPS URL Streamlit gives you.

If you want the OpenCV desktop build on your own PC, install from `requirements-desktop.txt` instead of the cloud file.

Option B: Your own server

1. Put `streamlit_app.py` on a host that can run Streamlit.
2. Serve it over HTTPS.
3. Share the public Streamlit URL.

Notes:

- Streamlit camera input works best over HTTPS.
- The app still stores employee, attendance, and model files under the project folders.
- If you use Streamlit Cloud or another hosted service, make sure file storage is persistent enough for your attendance records.

## What users get

- One browser link
- No app install
- Works on mobile and desktop
- Same employee and attendance data for everyone
