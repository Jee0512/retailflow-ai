# RetailFlow AI — Retail Store Analytics using CCTV

RetailFlow AI is a computer vision analytics platform that processes supermarket CCTV footage to track customer footfall, measure dwell time per zone, filter staff from shopper metrics, and generate operational reports — all without facial recognition or biometric identification.

## Features

- **YOLOv8 Person Detection** — Real-time object detection on CCTV footage with ByteTrack centroid tracking
- **Footfall Counting** — Cumulative shopper entry tracking across the store
- **Zone Dwell Analysis** — Per-aisle dwell time measurement and heatmap accumulation
- **Staff Filtering** — Uniform color-based staff detection to exclude associates from shopper analytics
- **Interactive Dashboard** — Streamlit-based real-time dashboard with KPI scorecards, charts, and video streaming
- **Home Portal** — Web landing page with authentication, admin console, and telemetry display
- **PDF Report Generation** — Automated store performance report with footfall, dwell, and zone metrics
- **SQLite Authentication** — User registration, login, and admin management with salted password hashing

## Technology Stack

- **Python 3.12**
- **YOLOv8 (Ultralytics)** — Object detection and tracking
- **OpenCV** — Video processing and frame analysis
- **Streamlit** — Interactive dashboard
- **Plotly** — Charts and visualizations
- **Pandas / NumPy** — Data processing
- **fpdf2** — PDF report generation
- **SQLite** — User authentication and admin management
- **http.server** — Portal HTTP server

## Project Structure

```
CV/
├── app.py                      # Streamlit dashboard (main entry point)
├── run.py                      # System launcher (deps, video, DB, portal, Streamlit)
├── portal_server.py            # HTTP portal server (port 8000)
├── auth_db.py                  # SQLite authentication and admin management
├── report_generator.py         # PDF report generation
├── start_system.bat            # Windows launcher batch script
├── requirements.txt            # Python dependencies
├── index.html                  # Home portal landing page
├── admin.html                  # Admin console
├── style.css / landing_style.css  # Stylesheets
├── utils/
│   ├── tracker_wrapper.py      # YOLOv8 tracking wrapper
│   ├── heatmap_generator.py    # Gaussian heatmap accumulation
│   └── synthetic_generator.py  # Synthetic CCTV video generation
├── docs/                       # Documentation and architecture files
├── assets/                     # Images and media
└── .streamlit/                 # Streamlit configuration
```

## Prerequisites

- Python 3.10+
- pip

## Installation

```powershell
# Clone the repository
git clone https://github.com/Jee0512/<repo-name>.git
cd <repo-name>

# Install dependencies
pip install -r requirements.txt

# (Optional) Install additional dependencies
pip install ultralytics streamlit opencv-python pandas matplotlib seaborn fpdf2 plotly
```

## Environment Variables

No external API keys are required. The application runs entirely locally.

## Running the Application

```powershell
python run.py
```

This will:
1. Check and install missing dependencies
2. Generate a sample CCTV video if needed
3. Initialize the SQLite authentication database
4. Start the portal server on port 8000
5. Launch the Streamlit dashboard on port 8501

Then open your browser:
- **Home Portal**: http://127.0.0.1:8000
- **Streamlit Dashboard**: http://localhost:8501
- **Admin Console**: http://127.0.0.1:8000/admin.html

## Usage

1. Open the Home Portal at `http://127.0.0.1:8000`
2. Click **Launch Main Dashboard** to open the Streamlit interface
3. Select **Simulation Demo (Sample Footage)** and click **Run Simulation Demo**
4. Watch real-time detection, tracking, and heatmap accumulation
5. Use the **Generate Store Report (PDF)** button to download a performance report

## Testing

```powershell
# Syntax check
python -m py_compile app.py run.py auth_db.py portal_server.py

# Verify core imports
python -c "import cv2, streamlit, ultralytics, pandas; print('OK')"
```

## Known Limitations

- Video processing is single-threaded; large videos may be slow
- Staff detection relies on uniform color matching (HSV ranges)
- No multi-camera support (single video feed at a time)
- Sample video is synthetic; real CCTV footage may require tuning

## Future Improvements

- Multi-camera feed support
- Real-time RTSP stream ingestion
- Improved staff detection using clothing segmentation
- Database-backed telemetry persistence
- Docker containerization

## License

No license specified. Contact the project author for usage terms.
```
filePath">
C:\Users\jeeva\Downloads\CV (2)\CV\README.md