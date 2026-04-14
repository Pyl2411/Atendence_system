# Deployment Guide for Daily Attendance System

## Overview
The Daily Attendance System is now a streamlined web application that opens the camera immediately upon loading, functioning like a biometric attendance machine. It features role-based access control and daily presence calculations.

## Prerequisites
- Python 3.8+
- Webcam/Camera access (required for face recognition)
- Internet connection (for location services)

## Installation

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

Required packages:
- numpy
- opencv-python
- streamlit
- Pillow

### 2. Setup Project Structure
Ensure the following directories exist:
- `data/` - Employee data and face samples
- `models/` - Trained face recognition models
- `attendance/` - Daily attendance CSV files
- `src/` - Core application modules

## Running the Application

### Development Mode
```powershell
# Windows
.\start_streamlit.ps1

# Or manually
streamlit run streamlit_app.py --server.port 8501 --server.address 0.0.0.0
```

### Production Deployment

#### Option 1: Streamlit Cloud
1. Push code to GitHub
2. Connect to Streamlit Cloud
3. Deploy with default settings

#### Option 2: Local Server
```bash
streamlit run streamlit_app.py --server.port 80 --server.address 0.0.0.0
```

#### Option 3: Docker
```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 8501

CMD ["streamlit", "run", "streamlit_app.py", "--server.address", "0.0.0.0"]
```

## Initial Setup

### 1. Register Employees
Use the desktop GUI application (`src/gui_app.py`) to:
1. Capture employee photos
2. Register employee details
3. Train the face recognition model

### 2. Train Model
```python
python -m src.train_model
```

### 3. Start Web App
The web app will automatically open the camera when users access it.

## User Roles & Permissions

- **Manager/Co Founder**: Full access to all attendance records and statistics
- **Employee/Team Leader/Trainee**: Access to personal attendance records only

## Features

- ✅ Immediate camera access on app load
- ✅ Face recognition for attendance marking
- ✅ Role-based access control
- ✅ Daily presence statistics
- ✅ Work hour calculations
- ✅ Location tracking (optional)
- ✅ Mobile-friendly interface
- ✅ No debug logs in production

## Security Considerations

- Camera access requires HTTPS in production
- Face data is stored locally in CSV files
- Consider database integration for production use
- Regular backups of attendance data recommended

## Troubleshooting

### Camera Not Working
- Ensure HTTPS in production environments
- Check browser camera permissions
- Verify OpenCV installation

### Face Not Recognized
- Retrain model with better lighting
- Ensure consistent face angles
- Check confidence threshold settings

### Performance Issues
- Reduce image processing resolution
- Implement database for large datasets
- Use cloud storage for face data

## API Endpoints (if needed)

The system uses CSV files for data storage. For production, consider:
- Database integration (PostgreSQL/MySQL)
- REST API development
- Cloud storage for images

## Support

For issues or questions:
- Check the logs in `data/capture_log.csv`
- Verify model files exist in `models/`
- Ensure proper file permissions

---
© 2024 Vickhardth Automation