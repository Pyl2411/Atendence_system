import cgi
import csv
import json
import os
import sys
import ssl
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

import cv2
import numpy as np


ROOT_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from capture_faces import append_capture_log, upsert_employee  # noqa: E402
from data_store import ATTENDANCE_DIR, DATA_DIR, load_attendance_records, load_employees  # noqa: E402
from mark_attendance import ATTENDANCE_COLUMNS, CONFIDENCE_THRESHOLD, mark_attendance  # noqa: E402
from train_model import LABELS_FILE, MODEL_FILE, MODELS_DIR, load_training_data  # noqa: E402


INDEX_HTML = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Attendance Mobile Web</title>
  <style>
    :root {
      --bg: #08111f;
      --panel: rgba(12, 22, 39, 0.92);
      --panel-2: rgba(17, 30, 50, 0.98);
      --text: #e8eef9;
      --muted: #a7b4c7;
      --accent: #5cd6b3;
      --accent-2: #6ea8fe;
      --danger: #ff6b6b;
      --line: rgba(255, 255, 255, 0.09);
      --shadow: 0 18px 60px rgba(0, 0, 0, 0.35);
      --radius: 22px;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: Inter, Segoe UI, Arial, sans-serif;
      color: var(--text);
      background:
        radial-gradient(circle at top left, rgba(92, 214, 179, 0.18), transparent 30%),
        radial-gradient(circle at top right, rgba(110, 168, 254, 0.16), transparent 25%),
        linear-gradient(180deg, #08111f 0%, #0b1628 45%, #09101b 100%);
      min-height: 100vh;
    }
    .wrap {
      max-width: 1220px;
      margin: 0 auto;
      padding: 18px;
    }
    .hero, .card {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: var(--radius);
      box-shadow: var(--shadow);
      backdrop-filter: blur(12px);
    }
    .hero {
      padding: 22px;
      display: grid;
      gap: 14px;
    }
    .title {
      display: flex;
      flex-wrap: wrap;
      justify-content: space-between;
      align-items: end;
      gap: 12px;
    }
    h1 {
      margin: 0;
      font-size: clamp(1.8rem, 4vw, 3rem);
      letter-spacing: -0.03em;
    }
    .sub {
      margin: 0;
      color: var(--muted);
      line-height: 1.5;
    }
    .pillrow {
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
    }
    .pill {
      padding: 8px 12px;
      border-radius: 999px;
      background: rgba(255, 255, 255, 0.06);
      border: 1px solid var(--line);
      color: var(--muted);
      font-size: 0.92rem;
    }
    .grid {
      display: grid;
      grid-template-columns: repeat(12, 1fr);
      gap: 16px;
      margin-top: 16px;
    }
    .card {
      padding: 18px;
      background: var(--panel-2);
    }
    .span-12 { grid-column: span 12; }
    .span-8 { grid-column: span 8; }
    .span-6 { grid-column: span 6; }
    .span-4 { grid-column: span 4; }
    .span-3 { grid-column: span 3; }
    .head {
      display: flex;
      justify-content: space-between;
      align-items: center;
      gap: 12px;
      margin-bottom: 14px;
    }
    .head h2 {
      margin: 0;
      font-size: 1.08rem;
    }
    .status {
      color: var(--muted);
      font-size: 0.95rem;
    }
    .status strong { color: var(--text); }
    .inputs {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 12px;
    }
    .inputs.one { grid-template-columns: 1fr; }
    label {
      display: grid;
      gap: 6px;
      font-size: 0.88rem;
      color: var(--muted);
    }
    input, select, textarea, button {
      font: inherit;
    }
    input, select, textarea {
      width: 100%;
      padding: 12px 14px;
      border-radius: 14px;
      border: 1px solid var(--line);
      background: rgba(255, 255, 255, 0.05);
      color: var(--text);
      outline: none;
    }
    input::placeholder, textarea::placeholder { color: #91a2bb; }
    .actions {
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
      margin-top: 14px;
    }
    button {
      border: 0;
      padding: 12px 16px;
      border-radius: 14px;
      cursor: pointer;
      font-weight: 700;
      color: #06111d;
      background: linear-gradient(135deg, var(--accent), #90f3cb);
      transition: transform 0.15s ease, filter 0.15s ease;
    }
    button.secondary {
      background: linear-gradient(135deg, var(--accent-2), #9ec1ff);
    }
    button.ghost {
      background: transparent;
      color: var(--text);
      border: 1px solid var(--line);
    }
    button.danger {
      background: linear-gradient(135deg, #ff7b7b, #ffb1b1);
    }
    button:active { transform: translateY(1px); filter: brightness(0.96); }
    button:disabled { opacity: 0.5; cursor: not-allowed; }
    .camera {
      display: grid;
      gap: 12px;
    }
    video, canvas.preview, img.thumb {
      width: 100%;
      border-radius: 18px;
      border: 1px solid var(--line);
      background: #020816;
      object-fit: cover;
    }
    video { aspect-ratio: 4 / 3; }
    .thumbs {
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(84px, 1fr));
      gap: 10px;
    }
    img.thumb {
      aspect-ratio: 1 / 1;
      height: 84px;
    }
    .list {
      display: grid;
      gap: 10px;
      max-height: 420px;
      overflow: auto;
      padding-right: 4px;
    }
    .row {
      padding: 12px;
      border-radius: 16px;
      border: 1px solid var(--line);
      background: rgba(255, 255, 255, 0.04);
    }
    .row strong { display: block; margin-bottom: 4px; }
    .small { color: var(--muted); font-size: 0.9rem; line-height: 1.4; }
    .topstats {
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 12px;
    }
    .stat {
      padding: 16px;
      border-radius: 18px;
      background: rgba(255, 255, 255, 0.05);
      border: 1px solid var(--line);
    }
    .stat .k { color: var(--muted); font-size: 0.84rem; }
    .stat .v { font-size: 1.8rem; font-weight: 800; margin-top: 4px; }
    .note {
      color: var(--muted);
      font-size: 0.94rem;
      line-height: 1.5;
    }
    .secure {
      padding: 12px 14px;
      border-radius: 16px;
      border: 1px solid var(--line);
      background: rgba(255, 255, 255, 0.04);
      color: var(--muted);
      line-height: 1.5;
    }
    .secure strong { color: var(--text); }
    @media (max-width: 960px) {
      .span-8, .span-6, .span-4, .span-3 { grid-column: span 12; }
      .topstats { grid-template-columns: 1fr; }
      .inputs { grid-template-columns: 1fr; }
    }
  </style>
</head>
<body>
  <div class="wrap">
    <section class="hero">
      <div class="title">
        <div>
          <h1>Attendance on Mobile</h1>
          <p class="sub">Open this link in any phone browser. Register employees, capture face photos, train the model, and mark attendance without installing anything.</p>
        </div>
        <div class="pillrow">
          <span class="pill" id="connPill">Connecting...</span>
          <span class="pill">Browser based</span>
          <span class="pill">Camera supported</span>
        </div>
      </div>
      <div class="topstats">
        <div class="stat"><div class="k">Employees</div><div class="v" id="statEmployees">0</div></div>
        <div class="stat"><div class="k">Attendance rows</div><div class="v" id="statAttendance">0</div></div>
        <div class="stat"><div class="k">Last status</div><div class="v" style="font-size:1rem" id="statStatus">Ready</div></div>
      </div>
      <div class="secure" id="secureBox">Checking browser security...</div>
    </section>

    <div class="grid">
      <section class="card span-6">
        <div class="head">
          <h2>Backend</h2>
          <div class="status" id="backendStatus">Waiting for server</div>
        </div>
        <div class="inputs one">
          <label>
            API base URL
            <input id="baseUrl" value="" placeholder="http://192.168.1.20:8000" />
          </label>
        </div>
        <div class="actions">
          <button class="secondary" id="btnConnect">Connect</button>
          <button class="ghost" id="btnRefresh">Refresh</button>
          <button id="btnTrain">Train Model</button>
        </div>
        <p class="note">If you are on the same Wi-Fi as the PC, enter the PC’s LAN IP here. On Android emulator, use <code>http://10.0.2.2:8000</code>.</p>
      </section>

      <section class="card span-6">
        <div class="head">
          <h2>Live Camera</h2>
          <div class="status" id="cameraStatus">Camera not started</div>
        </div>
        <div class="camera">
          <video id="video" autoplay playsinline muted></video>
          <canvas id="canvas" class="preview" style="display:none;"></canvas>
          <div class="actions">
            <button class="secondary" id="btnStartCam">Start Camera</button>
            <button class="ghost" id="btnStopCam">Stop</button>
            <button class="ghost" id="btnFlip">Use Front Camera</button>
          </div>
          <input id="sampleFile" type="file" accept="image/*" capture="environment" hidden />
          <input id="attendanceFile" type="file" accept="image/*" capture="environment" hidden />
          <p class="note">If live camera is blocked on your phone, the capture buttons will fall back to the phone camera picker.</p>
        </div>
      </section>

      <section class="card span-6">
        <div class="head">
          <h2>Register Employee</h2>
          <div class="status">Capture 1 or more samples</div>
        </div>
        <div class="inputs">
          <label>Name<input id="name" placeholder="John Smith" /></label>
          <label>Mobile<input id="mobile" placeholder="9876543210" /></label>
          <label>Employee ID<input id="employeeId" placeholder="EMP001" /></label>
          <label>Role
            <select id="role">
              <option>Employee</option>
              <option>Manager</option>
              <option>Co Founder</option>
              <option>Team Leader</option>
              <option>Trainee</option>
            </select>
          </label>
          <label>Company<input id="companyName" value="Vickhardth Automation" /></label>
          <label>Logo path optional<input id="logoPath" placeholder="" /></label>
        </div>
        <div class="actions">
          <button id="btnAddSample" class="secondary">Capture Sample</button>
          <button id="btnClearSamples" class="ghost">Clear Samples</button>
          <button id="btnRegister">Register Employee</button>
        </div>
        <div class="small" id="sampleInfo">0 samples captured</div>
        <div class="thumbs" id="sampleThumbs"></div>
      </section>

      <section class="card span-6">
        <div class="head">
          <h2>Mark Attendance</h2>
          <div class="status">Capture one photo then upload</div>
        </div>
        <div class="actions">
          <button id="btnCaptureAttendance" class="secondary">Capture Face</button>
          <button id="btnMarkAttendance">Mark Attendance</button>
          <button id="btnClearAttendance" class="ghost">Clear Photo</button>
        </div>
        <canvas id="attendancePreview" class="preview" style="display:none;"></canvas>
        <div class="small" id="attendanceInfo">No attendance photo yet</div>
      </section>

      <section class="card span-6">
        <div class="head">
          <h2>Employees</h2>
          <div class="status" id="employeesCount">0 rows</div>
        </div>
        <div class="list" id="employeesList"></div>
      </section>

      <section class="card span-6">
        <div class="head">
          <h2>Attendance</h2>
          <div class="status" id="attendanceCount">0 rows</div>
        </div>
        <div class="list" id="attendanceList"></div>
      </section>
    </div>
  </div>

  <script>
    const state = {
      baseUrl: "",
      stream: null,
      facingMode: "environment",
      samples: [],
      attendanceBlob: null,
    };

    const el = (id) => document.getElementById(id);
    const setStatus = (text) => {
      el("statStatus").textContent = text;
      el("backendStatus").textContent = text;
    };
    const setConnection = (ok) => {
      el("connPill").textContent = ok ? "Connected" : "Offline";
      el("connPill").style.color = ok ? "var(--accent)" : "var(--danger)";
    };

    function updateSecurityMessage() {
      if (window.isSecureContext) {
        el("secureBox").innerHTML = "<strong>Secure context detected.</strong> Camera access should work from HTTPS or localhost.";
      } else if (window.location.hostname === "localhost" || window.location.hostname === "127.0.0.1") {
        el("secureBox").innerHTML = "<strong>Localhost exception detected.</strong> Camera access should work on this PC, but public links need HTTPS.";
      } else {
        el("secureBox").innerHTML = "<strong>Not secure.</strong> Browser camera will usually fail until this site is served over HTTPS.";
      }
    }

    function resolveBaseUrl() {
      const value = el("baseUrl").value.trim();
      if (value) return value.replace(/\/$/, "");
      return window.location.origin;
    }

    async function api(path, options = {}) {
      const base = resolveBaseUrl();
      const response = await fetch(base + path, options);
      const text = await response.text();
      let data;
      try { data = text ? JSON.parse(text) : {}; } catch { data = { detail: text }; }
      if (!response.ok) {
        throw new Error(data.detail || response.statusText);
      }
      return data;
    }

    function drawVideoFrame(canvas) {
      const video = el("video");
      canvas.width = video.videoWidth || 1280;
      canvas.height = video.videoHeight || 720;
      const ctx = canvas.getContext("2d");
      ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
    }

    async function canvasToBlob(canvas) {
      return new Promise((resolve) => canvas.toBlob(resolve, "image/jpeg", 0.92));
    }

    async function handleCapturedFile(file, kind) {
      if (!file) return;
      if (kind === "sample") {
        state.samples.push(file);
        const url = URL.createObjectURL(file);
        const img = document.createElement("img");
        img.className = "thumb";
        img.src = url;
        img.onload = () => URL.revokeObjectURL(url);
        el("sampleThumbs").prepend(img);
        el("sampleInfo").textContent = `${state.samples.length} sample${state.samples.length === 1 ? "" : "s"} captured`;
        setStatus("Sample added from camera picker");
      } else {
        state.attendanceBlob = file;
        const url = URL.createObjectURL(file);
        const img = new Image();
        img.onload = () => {
          const canvas = el("attendancePreview");
          canvas.width = img.naturalWidth;
          canvas.height = img.naturalHeight;
          canvas.getContext("2d").drawImage(img, 0, 0);
          canvas.style.display = "block";
          URL.revokeObjectURL(url);
        };
        img.src = url;
        el("attendanceInfo").textContent = "Attendance photo ready";
        setStatus("Attendance photo added from camera picker");
      }
    }

    async function startCamera() {
      try {
        if (!window.isSecureContext && window.location.hostname !== "localhost" && window.location.hostname !== "127.0.0.1") {
          throw new Error("Camera requires HTTPS on public sites.");
        }
        if (state.stream) {
          state.stream.getTracks().forEach(t => t.stop());
        }
        state.stream = await navigator.mediaDevices.getUserMedia({
          video: { facingMode: state.facingMode },
          audio: false,
        });
        el("video").srcObject = state.stream;
        el("cameraStatus").textContent = "Camera ready";
      } catch (err) {
        el("cameraStatus").textContent = "Camera error: " + err.message;
      }
    }

    function stopCamera() {
      if (state.stream) {
        state.stream.getTracks().forEach(t => t.stop());
        state.stream = null;
      }
      el("video").srcObject = null;
      el("cameraStatus").textContent = "Camera stopped";
    }

    async function captureSample() {
      if (!state.stream) {
        el("sampleFile").click();
        return;
      }
      const canvas = el("canvas");
      drawVideoFrame(canvas);
      canvas.style.display = "block";
      const blob = await canvasToBlob(canvas);
      if (!blob) return;
      await handleCapturedFile(blob, "sample");
      setStatus("Sample captured");
    }

    async function captureAttendance() {
      if (!state.stream) {
        el("attendanceFile").click();
        return;
      }
      const canvas = el("attendancePreview");
      drawVideoFrame(canvas);
      canvas.style.display = "block";
      const blob = await canvasToBlob(canvas);
      await handleCapturedFile(blob, "attendance");
      setStatus("Attendance photo captured");
    }

    function clearSamples() {
      state.samples = [];
      el("sampleThumbs").innerHTML = "";
      el("sampleInfo").textContent = "0 samples captured";
      setStatus("Samples cleared");
    }

    function clearAttendance() {
      state.attendanceBlob = null;
      el("attendancePreview").style.display = "none";
      el("attendanceInfo").textContent = "No attendance photo yet";
      setStatus("Attendance photo cleared");
    }

    async function refreshData() {
      try {
        const health = await api("/health");
        const employees = await api("/employees");
        const attendance = await api("/attendance");

        el("statEmployees").textContent = health.employees ?? employees.length;
        el("statAttendance").textContent = health.attendance_rows ?? attendance.length;
        el("employeesCount").textContent = `${employees.length} rows`;
        el("attendanceCount").textContent = `${attendance.length} rows`;
        setConnection(true);
        setStatus("Connected");

        el("employeesList").innerHTML = employees.length
          ? employees.map(row => `
              <div class="row">
                <strong>${row.Name || ""}</strong>
                <div class="small">${row.EmployeeID || ""} | ${row.Role || ""}</div>
                <div class="small">${row.Mobile || ""} | ${row.CompanyName || ""}</div>
              </div>
            `).join("")
          : '<div class="row"><div class="small">No employees yet.</div></div>';

        el("attendanceList").innerHTML = attendance.length
          ? attendance.map(row => `
              <div class="row">
                <strong>${row.Name || ""}</strong>
                <div class="small">${row.Date || ""} | IN ${row.CheckIn || ""}</div>
                <div class="small">OUT ${row.CheckOut || ""} | Hours ${row.WorkHours || "-"}</div>
              </div>
            `).join("")
          : '<div class="row"><div class="small">No attendance rows yet.</div></div>';
      } catch (err) {
        setConnection(false);
        setStatus("Load failed: " + err.message);
      }
    }

    async function connect() {
      state.baseUrl = resolveBaseUrl();
      await refreshData();
    }

    async function trainModel() {
      try {
        setStatus("Training model...");
        await api("/train", { method: "POST" });
        setStatus("Model trained");
        await refreshData();
      } catch (err) {
        setStatus("Train failed: " + err.message);
      }
    }

    async function registerEmployee() {
      try {
        const fd = new FormData();
        fd.append("name", el("name").value.trim());
        fd.append("mobile", el("mobile").value.trim());
        fd.append("employee_id", el("employeeId").value.trim());
        fd.append("role", el("role").value.trim());
        fd.append("company_name", el("companyName").value.trim());
        fd.append("logo_path", el("logoPath").value.trim());
        state.samples.forEach((blob, index) => {
          fd.append("samples", blob, `sample_${index + 1}.jpg`);
        });
        setStatus("Registering employee...");
        const data = await api("/employees/register", { method: "POST", body: fd });
        setStatus(data.message + ` (${data.sample_count} samples)`);
        clearSamples();
        await refreshData();
      } catch (err) {
        setStatus("Register failed: " + err.message);
      }
    }

    async function markAttendance() {
      try {
        if (!state.attendanceBlob) {
          setStatus("Capture a face photo first");
          return;
        }
        const fd = new FormData();
        fd.append("file", state.attendanceBlob, "attendance.jpg");
        setStatus("Marking attendance...");
        const data = await api("/attendance/mark", { method: "POST", body: fd });
        setStatus(data.message);
        await refreshData();
      } catch (err) {
        setStatus("Attendance failed: " + err.message);
      }
    }

    el("btnStartCam").addEventListener("click", startCamera);
    el("btnStopCam").addEventListener("click", stopCamera);
    el("btnFlip").addEventListener("click", async () => {
      state.facingMode = state.facingMode === "environment" ? "user" : "environment";
      await startCamera();
      el("cameraStatus").textContent = "Camera switched";
    });
    el("btnConnect").addEventListener("click", connect);
    el("btnRefresh").addEventListener("click", refreshData);
    el("btnTrain").addEventListener("click", trainModel);
    el("btnAddSample").addEventListener("click", captureSample);
    el("btnClearSamples").addEventListener("click", clearSamples);
    el("btnRegister").addEventListener("click", registerEmployee);
    el("btnCaptureAttendance").addEventListener("click", captureAttendance);
    el("btnMarkAttendance").addEventListener("click", markAttendance);
    el("btnClearAttendance").addEventListener("click", clearAttendance);
    el("sampleFile").addEventListener("change", async (event) => {
      const file = event.target.files && event.target.files[0];
      await handleCapturedFile(file, "sample");
      event.target.value = "";
    });
    el("attendanceFile").addEventListener("change", async (event) => {
      const file = event.target.files && event.target.files[0];
      await handleCapturedFile(file, "attendance");
      event.target.value = "";
    });

    (async () => {
      el("baseUrl").value = window.location.origin;
      updateSecurityMessage();
      await refreshData();
    })();
  </script>
</body>
</html>
"""


def _ensure_parent_dirs():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    ATTENDANCE_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)


def _employee_folder(name, mobile):
    normalized_name = "_".join((name or "").strip().lower().split())
    normalized_mobile = "".join(ch for ch in (mobile or "") if ch.isdigit())
    return f"{normalized_name}_{normalized_mobile}" if normalized_mobile else normalized_name


def _preprocess_face(face_roi):
    resized = cv2.resize(face_roi, (200, 200))
    return cv2.equalizeHist(resized)


def _largest_face(gray_image):
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.2, minNeighbors=5)
    if len(faces) == 0:
        return None
    return max(faces, key=lambda rect: rect[2] * rect[3])


def _decode_image_bytes(raw_bytes):
    image = cv2.imdecode(np.frombuffer(raw_bytes, np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Invalid image data")
    return image


def _load_recognizer():
    if not MODEL_FILE.exists() or not LABELS_FILE.exists():
        raise FileNotFoundError("Model not trained yet.")

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read(str(MODEL_FILE))

    with LABELS_FILE.open("r", encoding="utf-8") as f:
        labels = {int(k): v for k, v in json.load(f).items()}

    return recognizer, labels


def _save_sample_images(name, mobile, file_items):
    folder_name = _employee_folder(name, mobile)
    person_dir = DATA_DIR / folder_name
    person_dir.mkdir(parents=True, exist_ok=True)

    saved = 0
    for item in file_items:
        raw = item["file"].read()
        if not raw:
            continue
        image = _decode_image_bytes(raw)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        face = _largest_face(gray)
        face_roi = gray if face is None else gray[face[1] : face[1] + face[3], face[0] : face[0] + face[2]]
        processed = _preprocess_face(face_roi)
        file_path = person_dir / f"{folder_name}_{saved:03d}.jpg"
        cv2.imwrite(str(file_path), processed)
        saved += 1

    return saved


def _recognize_image(raw_bytes):
    recognizer, labels = _load_recognizer()
    image = _decode_image_bytes(raw_bytes)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    face = _largest_face(gray)
    if face is None:
        raise ValueError("No face found in image")

    x, y, w, h = face
    face_roi = gray[y : y + h, x : x + w]
    processed = _preprocess_face(face_roi)
    label_id, confidence = recognizer.predict(processed)
    if confidence >= CONFIDENCE_THRESHOLD:
        return "unknown", confidence
    return labels.get(label_id, "unknown"), confidence


def _json_response(handler, status_code, payload):
    body = json.dumps(payload).encode("utf-8")
    handler.send_response(status_code)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(body)))
    handler.send_header("Access-Control-Allow-Origin", "*")
    handler.end_headers()
    handler.wfile.write(body)


def _read_post_data(handler):
    content_type = handler.headers.get("Content-Type", "")
    content_length = int(handler.headers.get("Content-Length", "0"))
    if content_length <= 0:
        return {}, []

    if "multipart/form-data" in content_type:
        form = cgi.FieldStorage(
            fp=handler.rfile,
            headers=handler.headers,
            environ={
                "REQUEST_METHOD": "POST",
                "CONTENT_TYPE": content_type,
                "CONTENT_LENGTH": str(content_length),
            },
            keep_blank_values=True,
        )
        fields = {}
        files = []
        for key in form.keys():
            item = form[key]
            if isinstance(item, list):
                for entry in item:
                    if getattr(entry, "filename", None):
                        files.append({"name": key, "filename": entry.filename, "file": entry.file})
                    else:
                        fields[key] = entry.value
            elif getattr(item, "filename", None):
                files.append({"name": key, "filename": item.filename, "file": item.file})
            else:
                fields[key] = item.value
        return fields, files

    raw = handler.rfile.read(content_length).decode("utf-8")
    parsed = dict(item.split("=", 1) for item in raw.split("&") if "=" in item)
    return parsed, []


class AttendanceHandler(BaseHTTPRequestHandler):
    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET,POST,OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path in ("/", "/index.html"):
            body = INDEX_HTML.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(body)
            return
        if parsed.path == "/health":
            _json_response(
                self,
                200,
                {
                    "status": "ok",
                    "employees": len(load_employees()),
                    "attendance_rows": len(load_attendance_records()),
                },
            )
            return
        if parsed.path == "/employees":
            _json_response(self, 200, load_employees())
            return
        if parsed.path == "/attendance":
            _json_response(self, 200, load_attendance_records())
            return

        _json_response(self, 404, {"detail": "Not found"})

    def do_POST(self):
        parsed = urlparse(self.path)
        fields, files = _read_post_data(self)

        try:
            if parsed.path == "/employees/register":
                name = (fields.get("name") or "").strip()
                mobile = (fields.get("mobile") or "").strip()
                employee_id = (fields.get("employee_id") or "").strip()
                role = (fields.get("role") or "").strip()
                company_name = (fields.get("company_name") or "").strip()
                logo_path = (fields.get("logo_path") or "").strip()

                if not name:
                    raise ValueError("Employee name is required.")
                if len("".join(ch for ch in mobile if ch.isdigit())) < 10:
                    raise ValueError("Valid mobile number is required.")
                if not employee_id:
                    raise ValueError("Employee ID is required.")
                if not role:
                    raise ValueError("Role is required.")
                if not company_name:
                    raise ValueError("Company name is required.")

                upsert_employee(name, mobile, employee_id, role, company_name, logo_path)
                saved = _save_sample_images(name, mobile, files)
                append_capture_log(name, mobile, _employee_folder(name, mobile), saved)

                _json_response(
                    self,
                    200,
                    {
                        "message": "Employee registered successfully.",
                        "employee_count": len(load_employees()),
                        "sample_count": saved,
                    },
                )
                return

            if parsed.path == "/train":
                images, labels, label_map = load_training_data()
                if not images:
                    raise ValueError("No training images available.")

                MODELS_DIR.mkdir(parents=True, exist_ok=True)
                recognizer = cv2.face.LBPHFaceRecognizer_create()
                recognizer.train(images, labels)
                recognizer.save(str(MODEL_FILE))
                with LABELS_FILE.open("w", encoding="utf-8") as f:
                    json.dump(label_map, f, indent=2)

                _json_response(
                    self,
                    200,
                    {
                        "message": "Model trained successfully.",
                        "model_file": str(MODEL_FILE),
                        "labels_file": str(LABELS_FILE),
                    },
                )
                return

            if parsed.path == "/attendance/mark":
                if not files:
                    raise ValueError("Image file is required.")
                name, confidence = _recognize_image(files[0]["file"].read())
                if name == "unknown":
                    _json_response(
                        self,
                        200,
                        {
                            "status": "rejected",
                            "name": None,
                            "confidence": confidence,
                            "message": "Face not recognized.",
                        },
                    )
                    return

                today_file = ATTENDANCE_DIR / f"attendance_{datetime.now().strftime('%Y%m%d')}.csv"
                ATTENDANCE_DIR.mkdir(parents=True, exist_ok=True)
                if not today_file.exists():
                    with today_file.open("w", newline="", encoding="utf-8") as f:
                        writer = csv.writer(f)
                        writer.writerow(ATTENDANCE_COLUMNS)

                message, marked = mark_attendance(name, today_file)
                _json_response(
                    self,
                    200,
                    {
                        "status": "marked" if marked else "blocked",
                        "name": name,
                        "confidence": confidence,
                        "message": message,
                    },
                )
                return

            _json_response(self, 404, {"detail": "Not found"})
        except Exception as exc:
            _json_response(self, 400, {"detail": str(exc)})


def main():
    _ensure_parent_dirs()
    port = int(os.environ.get("ATTENDANCE_PORT", "8000"))
    server = ThreadingHTTPServer(("0.0.0.0", port), AttendanceHandler)
    cert_file = os.environ.get("ATTENDANCE_CERT_FILE", "").strip()
    key_file = os.environ.get("ATTENDANCE_KEY_FILE", "").strip()
    if cert_file and key_file:
        context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        context.load_cert_chain(certfile=cert_file, keyfile=key_file)
        server.socket = context.wrap_socket(server.socket, server_side=True)
        scheme = "https"
    else:
        scheme = "http"
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
