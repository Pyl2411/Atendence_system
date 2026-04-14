# Start the Daily Attendance System Web App
# This script starts the Streamlit web application for face recognition attendance

# Set the Python path to include the src directory
$env:PYTHONPATH = "$PSScriptRoot\src"

# Change to the project directory
Set-Location $PSScriptRoot

# Check if virtual environment exists and activate it
if (Test-Path ".\.venv\Scripts\Activate.ps1") {
    & ".\.venv\Scripts\Activate.ps1"
}

# Start Streamlit app
streamlit run streamlit_app.py --server.port 8501 --server.address 0.0.0.0
