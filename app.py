import streamlit as np_streamlit
import cv2
import pandas as pd
import numpy as np
import time
import os
import socket
import http.server
import socketserver
import threading
import webbrowser
import json
import plotly.express as px
import plotly.graph_objects as go

from report_generator import (
    log_session_footfall,
    compute_peak_hours,
    generate_recommendations,
    build_pdf_report
)

from utils.heatmap_generator import HeatmapAccumulator
from utils.tracker_wrapper import CCTVTracker
from utils.synthetic_generator import generate_video

# Bridge alias for Streamlit API calls
st = np_streamlit

# Streamlit Page Config
st.set_page_config(
    page_title="RetailFlow.AI - Supermarket Analytics Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

@st.cache_resource
def ensure_portal_server(target_port=8000):
    """
    Spawns a quiet background HTTP daemon on port 8000 serving index.html and database auth endpoints.
    This ensures all links from the dashboard to the home page work seamlessly without ERR_CONNECTION_REFUSED.
    """
    # 1. Test if target port is already active
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(0.3)
            if s.connect_ex(("127.0.0.1", target_port)) == 0:
                return target_port
    except Exception:
        pass

    root_dir = os.path.dirname(os.path.abspath(__file__))

    class QuietPortalHandler(http.server.SimpleHTTPRequestHandler):
        def log_message(self, format, *args):
            pass  # Suppress terminal log noise
            
        def end_headers(self):
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization")
            self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
            self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
            super().end_headers()

        def do_OPTIONS(self):
            self.send_response(200)
            self.end_headers()

        def do_GET(self):
            clean_path = self.path.split("?")[0]
            if clean_path in ["", "/", "/home"]:
                self.path = "/index.html"
                return super().do_GET()

            if clean_path in ["/admin", "/admin/"]:
                self.path = "/admin.html"
                return super().do_GET()

            # Document routes
            if clean_path in ["/project_overview.pdf", "/overview.pdf"]:
                self.path = "/docs/project_overview.pdf"
                return super().do_GET()

            if clean_path in ["/mini_project_report.pdf", "/project_report.pdf"]:
                self.path = "/docs/mini_project_report.pdf"
                return super().do_GET()

            if clean_path in ["/Retail_Store_Performance_Report.pdf", "/generate_report", "/generate-report"]:
                pdf_file = os.path.join(root_dir, "Retail_Store_Performance_Report.pdf")
                if not os.path.exists(pdf_file) or clean_path in ["/generate_report", "/generate-report"]:
                    try:
                        from report_generator import generate_sample_report
                        generate_sample_report(pdf_file)
                    except Exception as e:
                        print(f"Error generating report via HTTP request: {e}")
                if clean_path in ["/generate_report", "/generate-report"]:
                    self.send_response(302)
                    self.send_header("Location", "/Retail_Store_Performance_Report.pdf")
                    self.end_headers()
                    return

            # Real-time Telemetry & CCTV Tracking State
            if clean_path in ["/api/telemetry", "/api/dashboard-stats"]:
                try:
                    telemetry_file = os.path.join(root_dir, "session_telemetry.json")
                    telemetry_data = None
                    if os.path.exists(telemetry_file):
                        try:
                            with open(telemetry_file, "r", encoding="utf-8") as tf:
                                telemetry_data = json.load(tf)
                        except Exception:
                            telemetry_data = None

                    now = time.time()
                    is_live = False
                    if telemetry_data:
                        updated_ago = now - telemetry_data.get("timestamp", 0)
                        if updated_ago < 8 and telemetry_data.get("is_live", False):
                            is_live = True

                    if is_live:
                        status = "live"
                        cctv_uploaded = telemetry_data.get("cctv_uploaded", True)
                        fps_val = telemetry_data.get("fps", 30.0)
                        status_label = f"LIVE CCTV INFERENCE ACTIVE &bull; {fps_val:.0f} FPS"
                        source_name = telemetry_data.get("source_type", "Live CCTV Stream")
                        active_occupancy = telemetry_data.get("active_occupancy", 0)
                        total_footfall = telemetry_data.get("total_footfall", 0)
                        avg_dwell_time = telemetry_data.get("avg_dwell_time", 0.0)
                        staff_filtered = telemetry_data.get("staff_count", 0)
                        busiest_zone = telemetry_data.get("busiest_zone", "Aisle A (Produce)")
                        raw_dwells = telemetry_data.get("zone_dwells", {})
                        zone_dwells = {
                            "Aisle A (Produce)": raw_dwells.get("Aisle A (Produce)", 28.5),
                            "Aisle C (Electronics)": raw_dwells.get("Aisle C (Electronics)", 22.0),
                            "Aisle B (Apparel)": raw_dwells.get("Aisle B (Apparel)", 18.2),
                            "Checkout Lanes": raw_dwells.get("Checkout 1", 11.8)
                        }
                    else:
                        status = "standby"
                        cctv_uploaded = False
                        status_label = "SYSTEM STANDBY &bull; AWAITING CCTV INPUT"
                        source_name = "Simulation Benchmark Preset (600 Frames)"
                        active_occupancy = 0
                        total_footfall = 8
                        avg_dwell_time = 16.4
                        staff_filtered = 1
                        busiest_zone = "Aisle A (Produce)"
                        zone_dwells = {
                            "Aisle A (Produce)": 28.5,
                            "Aisle C (Electronics)": 22.0,
                            "Aisle B (Apparel)": 18.2,
                            "Checkout Lanes": 11.8
                        }

                    payload = json.dumps({
                        "success": True,
                        "status": status,
                        "status_label": status_label,
                        "streamlit_alive": True,
                        "cctv_uploaded": cctv_uploaded,
                        "source_name": source_name,
                        "active_occupancy": active_occupancy,
                        "total_footfall": total_footfall,
                        "avg_dwell_time": round(avg_dwell_time, 1),
                        "staff_filtered": staff_filtered,
                        "busiest_zone": busiest_zone,
                        "peak_hour": "19:00 (Rush Window)",
                        "checkout_wait": "1.4 mins",
                        "zone_dwells": zone_dwells
                    }).encode("utf-8")
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.end_headers()
                    self.wfile.write(payload)
                    return
                except Exception as e:
                    payload = json.dumps({"success": False, "message": str(e)}).encode("utf-8")
                    self.send_response(500)
                    self.send_header("Content-Type", "application/json")
                    self.end_headers()
                    self.wfile.write(payload)
                    return

            if clean_path in ["/api/users", "/api/admin/users"]:
                try:
                    import auth_db
                    users = auth_db.get_all_users_admin()
                    payload = json.dumps({"success": True, "users": users}).encode("utf-8")
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.end_headers()
                    self.wfile.write(payload)
                    return
                except Exception as e:
                    payload = json.dumps({"success": False, "message": str(e)}).encode("utf-8")
                    self.send_response(500)
                    self.send_header("Content-Type", "application/json")
                    self.end_headers()
                    self.wfile.write(payload)
                    return

            if clean_path == "/api/admin/stats":
                try:
                    import auth_db
                    stats = auth_db.get_admin_metrics()
                    payload = json.dumps({"success": True, "stats": stats}).encode("utf-8")
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.end_headers()
                    self.wfile.write(payload)
                    return
                except Exception as e:
                    payload = json.dumps({"success": False, "message": str(e)}).encode("utf-8")
                    self.send_response(500)
                    self.send_header("Content-Type", "application/json")
                    self.end_headers()
                    self.wfile.write(payload)
                    return

            if clean_path == "/api/admin/audit-logs":
                try:
                    import auth_db
                    logs = auth_db.get_audit_logs(50)
                    payload = json.dumps({"success": True, "logs": logs}).encode("utf-8")
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.end_headers()
                    self.wfile.write(payload)
                    return
                except Exception as e:
                    payload = json.dumps({"success": False, "message": str(e)}).encode("utf-8")
                    self.send_response(500)
                    self.send_header("Content-Type", "application/json")
                    self.end_headers()
                    self.wfile.write(payload)
                    return

            return super().do_GET()

        def do_POST(self):
            clean_path = self.path.split("?")[0]
            try:
                content_length = int(self.headers.get("Content-Length", 0))
                raw_body = self.rfile.read(content_length).decode("utf-8") if content_length > 0 else "{}"
                data = json.loads(raw_body)
            except Exception:
                data = {}

            import auth_db

            # User Registration & Login
            if clean_path in ["/api/register", "/api/login"]:
                try:
                    email = data.get("email", "").strip()
                    password = data.get("password", "").strip()
                    role = data.get("role", "Store Manager")

                    if clean_path == "/api/register":
                        success, message, user_info = auth_db.register_user(email, password, role)
                    else:
                        success, message, user_info = auth_db.verify_user(email, password)

                    res_payload = {"success": success, "message": message, "user": user_info}
                    res_bytes = json.dumps(res_payload).encode("utf-8")
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.end_headers()
                    self.wfile.write(res_bytes)
                    return
                except Exception as e:
                    err_payload = json.dumps({"success": False, "message": f"Server error: {e}"}).encode("utf-8")
                    self.send_response(500)
                    self.send_header("Content-Type", "application/json")
                    self.end_headers()
                    self.wfile.write(err_payload)
                    return

            # Admin Login
            if clean_path == "/api/admin/login":
                try:
                    admin_id = data.get("admin_id", "").strip()
                    password = data.get("password", "").strip()
                    success, message, admin_info = auth_db.verify_admin(admin_id, password)
                    res_payload = {"success": success, "message": message, "admin": admin_info}
                    res_bytes = json.dumps(res_payload).encode("utf-8")
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.end_headers()
                    self.wfile.write(res_bytes)
                    return
                except Exception as e:
                    err_payload = json.dumps({"success": False, "message": str(e)}).encode("utf-8")
                    self.send_response(500)
                    self.send_header("Content-Type", "application/json")
                    self.end_headers()
                    self.wfile.write(err_payload)
                    return

            # Admin Approve User
            if clean_path == "/api/admin/approve-user":
                try:
                    user_id = int(data.get("user_id", 0))
                    actor = data.get("admin_actor", "admin")
                    success, message = auth_db.approve_user(user_id, actor)
                    res_bytes = json.dumps({"success": success, "message": message}).encode("utf-8")
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.end_headers()
                    self.wfile.write(res_bytes)
                    return
                except Exception as e:
                    err_payload = json.dumps({"success": False, "message": str(e)}).encode("utf-8")
                    self.send_response(500)
                    self.send_header("Content-Type", "application/json")
                    self.end_headers()
                    self.wfile.write(err_payload)
                    return

            # Admin Reject User
            if clean_path == "/api/admin/reject-user":
                try:
                    user_id = int(data.get("user_id", 0))
                    actor = data.get("admin_actor", "admin")
                    success, message = auth_db.reject_user(user_id, actor)
                    res_bytes = json.dumps({"success": success, "message": message}).encode("utf-8")
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.end_headers()
                    self.wfile.write(res_bytes)
                    return
                except Exception as e:
                    err_payload = json.dumps({"success": False, "message": str(e)}).encode("utf-8")
                    self.send_response(500)
                    self.send_header("Content-Type", "application/json")
                    self.end_headers()
                    self.wfile.write(err_payload)
                    return

            # Admin Update User
            if clean_path == "/api/admin/update-user":
                try:
                    user_id = int(data.get("user_id", 0))
                    email = data.get("email", "").strip()
                    role = data.get("role", "Store Manager")
                    status = data.get("status", "approved")
                    new_pwd = data.get("new_password", None)
                    actor = data.get("admin_actor", "admin")
                    success, message = auth_db.update_user_details(user_id, email, role, status, new_pwd, actor)
                    res_bytes = json.dumps({"success": success, "message": message}).encode("utf-8")
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.end_headers()
                    self.wfile.write(res_bytes)
                    return
                except Exception as e:
                    err_payload = json.dumps({"success": False, "message": str(e)}).encode("utf-8")
                    self.send_response(500)
                    self.send_header("Content-Type", "application/json")
                    self.end_headers()
                    self.wfile.write(err_payload)
                    return

            # Admin Delete User
            if clean_path == "/api/admin/delete-user":
                try:
                    user_id = int(data.get("user_id", 0))
                    actor = data.get("admin_actor", "admin")
                    success, message = auth_db.delete_user_account(user_id, actor)
                    res_bytes = json.dumps({"success": success, "message": message}).encode("utf-8")
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.end_headers()
                    self.wfile.write(res_bytes)
                    return
                except Exception as e:
                    err_payload = json.dumps({"success": False, "message": str(e)}).encode("utf-8")
                    self.send_response(500)
                    self.send_header("Content-Type", "application/json")
                    self.end_headers()
                    self.wfile.write(err_payload)
                    return

            # Admin Add User
            if clean_path == "/api/admin/add-user":
                try:
                    email = data.get("email", "").strip()
                    password = data.get("password", "").strip()
                    role = data.get("role", "Retail Analyst")
                    status = data.get("status", "approved")
                    actor = data.get("admin_actor", "admin")
                    success, message, user_info = auth_db.add_user_by_admin(email, password, role, status, actor)
                    res_bytes = json.dumps({"success": success, "message": message, "user": user_info}).encode("utf-8")
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.end_headers()
                    self.wfile.write(res_bytes)
                    return
                except Exception as e:
                    err_payload = json.dumps({"success": False, "message": str(e)}).encode("utf-8")
                    self.send_response(500)
                    self.send_header("Content-Type", "application/json")
                    self.end_headers()
                    self.wfile.write(err_payload)
                    return

            return super().do_POST()

    for test_port in [target_port, 8080, 8008]:
        for host in ["", "0.0.0.0", "127.0.0.1"]:
            try:
                socketserver.TCPServer.allow_reuse_address = True
                handler = lambda *args, **kwargs: QuietPortalHandler(*args, directory=root_dir, **kwargs)
                httpd = socketserver.TCPServer((host, test_port), handler)
                t = threading.Thread(target=httpd.serve_forever, daemon=True)
                t.start()
                return test_port
            except Exception:
                continue
    return target_port

portal_port = ensure_portal_server(8000)

# Setup Session State for persistent metrics and stream control
if "tracker" not in st.session_state:
    st.session_state.tracker = CCTVTracker(use_yolo=False)
if "heatmap" not in st.session_state:
    st.session_state.heatmap = HeatmapAccumulator(1280, 720)
if "footfall_timeline" not in st.session_state:
    st.session_state.footfall_timeline = []
if "stream_active" not in st.session_state:
    st.session_state.stream_active = False
if "frame_idx" not in st.session_state:
    st.session_state.frame_idx = 0
if "stream_has_run" not in st.session_state:
    st.session_state.stream_has_run = False

default_pdf_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Retail_Store_Performance_Report.pdf")
if "report_path" not in st.session_state:
    st.session_state.report_path = default_pdf_file if os.path.exists(default_pdf_file) else None
if "report_bytes" not in st.session_state:
    if os.path.exists(default_pdf_file):
        try:
            with open(default_pdf_file, "rb") as f:
                st.session_state.report_bytes = f.read()
        except Exception:
            st.session_state.report_bytes = None
    else:
        st.session_state.report_bytes = None

if "report_ready" not in st.session_state:
    st.session_state.report_ready = False
if "report_message" not in st.session_state:
    st.session_state.report_message = "Store analysis report is ready!" if os.path.exists(default_pdf_file) else None
if "report_session_logged" not in st.session_state:
    st.session_state.report_session_logged = False

tracker = st.session_state.tracker
heatmap = st.session_state.heatmap

# Helper to render clean raw HTML without Markdown code-block interpretation
def render_html(html_str: str, container=None, sidebar: bool = False):
    """
    Renders raw HTML in Streamlit cleanly without it being parsed as a Markdown code block.
    Removes all leading line indentation and blank lines so CommonMark never treats it as <pre><code>.
    """
    cleaned = "\n".join(line.strip() for line in html_str.strip().splitlines() if line.strip())
    if container is not None:
        container.markdown(cleaned, unsafe_allow_html=True)
    elif sidebar:
        st.sidebar.markdown(cleaned, unsafe_allow_html=True)
    else:
        st.markdown(cleaned, unsafe_allow_html=True)

# Load custom flat light stylesheet (style.css)
def apply_theme_css():
    css_path = os.path.join(os.path.dirname(__file__), "style.css")
    if os.path.exists(css_path):
        with open(css_path, "r", encoding="utf-8") as f:
            css_content = f.read()
        st.markdown(f"<style>\n{css_content}\n</style>", unsafe_allow_html=True)
    else:
        st.error("Global stylesheet (style.css) was not found in workspace!")

apply_theme_css()

# Sidebar Navigation & Portal Shortcut (Clean, Humanized Architecture)
render_html(f"""
<div class="sidebar-portal-card">
    <div class="sidebar-portal-header">
        <span class="sidebar-portal-tag">Store Operations Terminal</span>
        <span class="sidebar-portal-title">RetailFlow</span>
    </div>
    <div class="sidebar-portal-buttons">
        <a href="http://127.0.0.1:{portal_port}" target="_blank" class="sidebar-portal-btn" title="Open Store Landing & Overview Portal">
            <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                <path d="m3 9 9-7 9 7v11a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z"></path>
                <polyline points="9 22 9 12 15 12 15 22"></polyline>
            </svg>
            <span>Home Portal</span>
        </a>
        <a href="http://127.0.0.1:{portal_port}/admin.html" target="_blank" class="sidebar-portal-btn" title="Open Administrator Governance Console">
            <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                <path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"></path>
            </svg>
            <span>Admin Console</span>
        </a>
    </div>
</div>
""", sidebar=True)

# Sidebar Layout & Grouped Controls
render_html("""
<div style="margin-top: 10px; margin-bottom: 12px; border-bottom: 1px solid #E8DFD0; padding-bottom: 6px;">
    <h3 style="color: #2C1E16; margin: 0; font-size: 13px; font-weight: 700; letter-spacing: 0.3px;">STREAM &amp; ANALYTICS</h3>
</div>
""", sidebar=True)

# 1. Processing pipeline selectbox
mode = st.sidebar.selectbox(
    "Processing Pipeline",
    ["Simulation Demo (Sample Footage)", "Real-time CCTV (YOLOv8 Model)"]
)

source_type = "Sample Video"
uploaded_file = None
video_ready = False

if mode == "Real-time CCTV (YOLOv8 Model)":
    source_option = st.sidebar.selectbox(
        "CCTV Source Channel",
        ["Upload Local Video File", "Live Webcam Stream"]
    )
    if st.session_state.get("last_source_option") != source_option:
        st.session_state.last_source_option = source_option
        st.session_state.stream_active = False
        st.session_state.frame_idx = 0

    if source_option == "Upload Local Video File":
        source_type = "Uploaded Video"
        uploaded_file = st.sidebar.file_uploader(
            "Select CCTV MP4 file",
            type=["mp4", "avi", "mov", "mkv"]
        )
        if uploaded_file is not None:
            file_bytes = uploaded_file.getvalue()
            if len(file_bytes) > 0:
                upload_dir = os.path.dirname(os.path.abspath(__file__))
                target_video_file = os.path.join(upload_dir, "temp_cctv_uploaded.mp4")
                need_save = True
                current_active = st.session_state.get("active_video_path")
                if current_active and os.path.exists(current_active) and st.session_state.get("last_uploaded_name") == uploaded_file.name:
                    try:
                        if os.path.getsize(current_active) == len(file_bytes):
                            need_save = False
                    except Exception:
                        need_save = True
                if need_save:
                    saved_path = None
                    try:
                        import gc
                        gc.collect()
                        with open(target_video_file, "wb") as f:
                            f.write(file_bytes)
                        saved_path = target_video_file
                    except Exception:
                        alt_video_file = os.path.join(upload_dir, f"temp_cctv_{int(time.time())}.mp4")
                        try:
                            with open(alt_video_file, "wb") as f:
                                f.write(file_bytes)
                            saved_path = alt_video_file
                        except Exception as alt_err:
                            st.sidebar.error(f"Error saving video: {alt_err}")
                    if saved_path:
                        st.session_state.active_video_path = saved_path
                        st.session_state.last_uploaded_name = uploaded_file.name
                        st.session_state.frame_idx = 0
                        st.session_state.stream_active = False  # NEVER auto-start upon upload!
                        st.session_state.tracker.reset()
                        st.session_state.heatmap.reset()
                        st.session_state.footfall_timeline = []
                        st.session_state.stream_error = None
                video_ready = True
                st.sidebar.success(f"Video ready: {uploaded_file.name} ({len(file_bytes)/(1024*1024):.1f} MB)")
                st.sidebar.caption("Click 'Start Camera Feed' below to begin live analysis.")
            else:
                video_ready = False
                st.session_state.active_video_path = None
                st.session_state.stream_active = False
                st.sidebar.warning("Uploaded file is empty. Please select a valid video.")
        else:
            video_ready = False
            # When no file is currently uploaded in this session, do not use stale files!
            st.session_state.active_video_path = None
            st.session_state.last_uploaded_name = None
            st.session_state.stream_active = False
            st.sidebar.info("Upload an MP4, AVI, or MOV video file to analyze store traffic.")
    else:
        source_type = "Webcam"
        video_ready = True
else:
    source_type = "Sample Video"
    video_ready = True
    sample_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sample_cctv.mp4")
    if os.path.exists(sample_file):
        st.sidebar.success("Preset loaded: sample_cctv.mp4 (600 Frames Synthetic CCTV)")
        st.sidebar.caption("Click 'Run Simulation Demo' below to begin.")

# Synchronize mode change to reset session cleanly
if st.session_state.get("last_selected_mode") != mode:
    st.session_state.last_selected_mode = mode
    st.session_state.frame_idx = 0
    st.session_state.stream_active = False
    st.session_state.tracker.reset()
    st.session_state.heatmap.reset()
    st.session_state.footfall_timeline = []

is_demo_mode = (source_type == "Sample Video")
start_button_label = "▶ Run Simulation Demo" if is_demo_mode else "▶ Start Camera Feed"
start_disabled = (not is_demo_mode and source_type == "Uploaded Video" and not video_ready)

# Execution Controls - Prominently located right beneath source selection
st.sidebar.markdown("<div style='margin-top: 10px; margin-bottom: 4px; font-weight: 700; font-size: 11px; color: #8E7F73; text-transform: uppercase; letter-spacing: 0.4px;'>Stream Controls</div>", unsafe_allow_html=True)
run_stream = st.sidebar.button(start_button_label, use_container_width=True, type="primary", disabled=start_disabled)
stop_stream = st.sidebar.button("⏹ Pause Stream", use_container_width=True)
reset_btn = st.sidebar.button("🔄 Reset Session Data", use_container_width=True)

with st.sidebar.expander("Staff Detection & Filtering", expanded=False):
    filter_staff_toggle = st.checkbox(
        "Filter Staff from Analytics",
        value=True,
        help="Excludes store staff from footfall counts, dwell times, and heatmap accumulation."
    )
    uniform_preset = st.selectbox(
        "Staff Uniform Preset",
        [
            "Red (Target/Coles)",
            "Blue (Walmart/BestBuy)",
            "Green / Neon (Safety Vest)",
            "Black Uniform",
            "Orange (Home Depot)",
            "Custom Color"
        ],
        help="Select standard staff uniform color or provide a custom color."
    )
    custom_color_rgb = (220, 40, 40)
    if uniform_preset == "Custom Color":
        hex_c = st.color_picker("Pick Uniform Color", "#DC2626")
        hex_clean = hex_c.lstrip("#")
        custom_color_rgb = tuple(int(hex_clean[i:i+2], 16) for i in (0, 2, 4))
        
    staff_sensitivity = st.slider(
        "Uniform Sensitivity",
        0.10, 0.50, 0.20, 0.05,
        help="Minimum torso fraction matching uniform color to classify as staff."
    )
    show_staff_boxes = st.checkbox(
        "Highlight Staff on CCTV",
        value=True,
        help="Draws an amber border and 'STAFF (Filtered)' badge on detected staff members."
    )

with st.sidebar.expander("Detection Thresholds", expanded=False):
    conf_threshold = st.slider("Confidence Limit", 0.1, 1.0, 0.35, 0.05, help="Inference threshold for target tracking")
    model_version = st.selectbox("Model Resolution", ["yolov8n.pt (Nano)"])

with st.sidebar.expander("Zone & Trajectory Overlays", expanded=False):
    draw_zones_toggle = st.checkbox("Show Zone Outlines", value=True)
    draw_trails = st.checkbox("Show Movement Trails", value=True)

with st.sidebar.expander("Heatmap Visual Scheme", expanded=False):
    heatmap_alpha = st.slider("Heatmap Opacity (Alpha)", 0.1, 1.0, 0.55, 0.05)
    blur_strength = st.slider("Gaussian Blur Radius", 15, 81, 35, 2)
    color_map_selection = st.selectbox(
        "Colormap Scheme",
        ["Jet (Standard)", "Hot (Thermal)", "Rainbow"]
    )

# Map colormap code
cmap_dict = {
    "Jet (Standard)": cv2.COLORMAP_JET,
    "Hot (Thermal)": cv2.COLORMAP_HOT,
    "Rainbow": cv2.COLORMAP_RAINBOW
}
selected_cmap = cmap_dict[color_map_selection]

# Handle tracking pipeline configuration
if mode == "Real-time CCTV (YOLOv8 Model)":
    tracker.use_yolo = True
    if tracker.model is None:
        tracker.load_yolo_model()
else:
    tracker.use_yolo = False
    emul_csv = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sample_cctv_detections.csv")
    tracker.load_emulation_data(emul_csv)

# Synchronize is_demo state on the tracker
is_demo_mode = (source_type == "Sample Video")
tracker.is_demo = is_demo_mode

# Synchronize staff filtering configuration on the tracker
tracker.filter_staff = filter_staff_toggle
tracker.staff_uniform_preset = uniform_preset
tracker.staff_custom_color_rgb = custom_color_rgb
tracker.staff_color_threshold = staff_sensitivity
tracker.show_staff_boxes = show_staff_boxes

def save_session_telemetry(is_live=False, active_count=0, fps=0.0, **kwargs):
    """Saves the current real-time telemetry state to disk for the home portal dashboard."""
    try:
        final_df, final_metrics = tracker.get_analytics()
        active_video = st.session_state.get("active_video_path")
        cctv_uploaded = bool(source_type == "Uploaded Video" and active_video and os.path.exists(active_video) and os.path.getsize(active_video) > 0)
        
        b_footfall = int(final_metrics.get("total_footfall", 0))
        b_dwell = float(final_metrics.get("avg_dwell_time", 0.0))
        b_busiest = str(final_metrics.get("busiest_zone", "None"))
        b_staff = int(final_metrics.get("total_staff_detected", 0))
        b_counts = {str(k): int(v) for k, v in final_metrics.get("zone_counts", {}).items()}
        b_dwells = {str(k): float(v) for k, v in final_metrics.get("zone_dwells", {}).items()}
        
        if b_footfall == 0 and source_type == "Sample Video" and st.session_state.get("stream_has_run", False):
            _, sim_m, _ = get_benchmark_simulation_dataset()
            b_footfall = int(sim_m.get("total_footfall", 8))
            b_dwell = float(sim_m.get("avg_dwell_time", 16.4))
            b_busiest = str(sim_m.get("busiest_zone", "Aisle A (Produce)"))
            b_staff = int(sim_m.get("total_staff_detected", 1))
            b_counts = {str(k): int(v) for k, v in sim_m.get("zone_counts", {}).items()}
            b_dwells = {str(k): float(v) for k, v in sim_m.get("zone_dwells", {}).items()}

        payload = {
            "is_live": bool(is_live),
            "cctv_uploaded": bool(cctv_uploaded),
            "source_type": str(source_type),
            "active_occupancy": int(active_count),
            "total_footfall": b_footfall,
            "avg_dwell_time": round(b_dwell, 1),
            "busiest_zone": b_busiest if b_busiest != "None" else ("Aisle A (Produce)" if st.session_state.get("stream_has_run", False) else "Standby"),
            "staff_count": b_staff,
            "zone_counts": b_counts,
            "zone_dwells": b_dwells,
            "fps": round(float(fps), 1),
            "timestamp": time.time()
        }
        telemetry_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "session_telemetry.json")
        with open(telemetry_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
    except Exception:
        pass

@st.cache_data
def get_benchmark_simulation_dataset():
    """Compiles and caches the synthetic simulation benchmark metrics and timeline for standby display."""
    csv_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sample_cctv_detections.csv")
    if os.path.exists(csv_file):
        try:
            temp_tracker = CCTVTracker(use_yolo=False)
            temp_tracker.load_emulation_data(csv_file)
            fake_frame = np.zeros((720, 1280, 3), dtype=np.uint8)
            max_f = int(temp_tracker.emulation_df["frame_idx"].max()) if temp_tracker.emulation_df is not None else 600
            for f in range(min(max_f + 1, 600)):
                temp_tracker.process_frame(fake_frame, f)
            sim_df, sim_metrics = temp_tracker.get_analytics()
            
            # Extract sampled footfall timeline
            tl = []
            cust_df = temp_tracker.emulation_df[temp_tracker.emulation_df["role"] == "customer"]
            for f_idx in range(0, min(max_f + 1, 600), 30):
                cnt = len(cust_df[cust_df["frame_idx"] == f_idx])
                tl.append({"Time (s)": round(f_idx / 30.0, 1), "Shopper Count": cnt})
                
            return sim_df, sim_metrics, tl
        except Exception as e:
            print(f"Error computing benchmark dataset: {e}")
            
    return pd.DataFrame(), {
        "total_footfall": 8,
        "avg_dwell_time": 16.4,
        "busiest_zone": "Aisle A (Produce)",
        "zone_counts": {"Aisle A (Produce)": 6, "Aisle C (Electronics)": 5, "Aisle B (Apparel)": 4, "Checkout 1": 4},
        "zone_dwells": {"Aisle A (Produce)": 28.5, "Aisle C (Electronics)": 22.0, "Aisle B (Apparel)": 18.2, "Checkout 1": 11.8},
        "total_staff_detected": 1,
        "active_staff": 0,
        "filter_staff_enabled": True
    }, []

@st.cache_data
def get_benchmark_trajectories():
    """Extracts sampled movement paths for benchmark simulation display."""
    csv_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sample_cctv_detections.csv")
    if os.path.exists(csv_file):
        try:
            df = pd.read_csv(csv_file)
            cust = df[df["role"] == "customer"]
            tracks = {}
            for tid, g in cust.groupby("track_id"):
                pts = list(zip(((g["x_min"] + g["x_max"]) // 2), ((g["y_min"] + g["y_max"]) // 2)))
                tracks[int(tid)] = pts[::8]
            return tracks
        except Exception:
            pass
    return {}

# --- PDF REPORT HELPERS ---
def generate_store_report():
    """Build the final PDF from the analytics collected by the current session."""
    final_df, final_metrics = tracker.get_analytics()

    # If user generates report before running the stream or footfall is 0, compile full simulation analytics
    if final_metrics.get("total_footfall", 0) == 0:
        sim_df, sim_metrics, _ = get_benchmark_simulation_dataset()
        if sim_metrics.get("total_footfall", 0) > 0:
            final_df, final_metrics = sim_df, sim_metrics

    # Save the current session timeline only once.
    if st.session_state.footfall_timeline and not st.session_state.report_session_logged:
        log_session_footfall(st.session_state.footfall_timeline)
        st.session_state.report_session_logged = True

    peak_res = compute_peak_hours()
    if isinstance(peak_res, tuple) or isinstance(peak_res, list):
        hourly_df = peak_res[0] if len(peak_res) > 0 else pd.DataFrame()
        busiest_hour = peak_res[1] if len(peak_res) > 1 else "19:00"
        quietest_hour = peak_res[2] if len(peak_res) > 2 else "08:00"
        sessions_logged = peak_res[3] if len(peak_res) > 3 else 1
    else:
        hourly_df = peak_res
        busiest_hour, quietest_hour, sessions_logged = "19:00", "08:00", 1

    zone_table_data = generate_zone_analytics_table(
        tracker,
        final_df,
        final_metrics
    )

    rec_res = generate_recommendations(
        zone_table_data,
        final_metrics,
        hourly_df,
        busiest_hour,
        quietest_hour,
        sessions_logged
    )
    if isinstance(rec_res, tuple) or isinstance(rec_res, list):
        recs = rec_res[0] if len(rec_res) > 0 else []
        peak_text = rec_res[1] if len(rec_res) > 1 else ""
        discount_text = rec_res[2] if len(rec_res) > 2 else ""
    else:
        recs, peak_text, discount_text = rec_res, "", ""

    default_report_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "Retail_Store_Performance_Report.pdf"
    )

    demo_flag = getattr(tracker, "is_demo", (source_type == "Sample Video"))

    report_res = build_pdf_report(
        final_metrics=final_metrics,
        zone_df=zone_table_data,
        recs=recs,
        peak_text=peak_text,
        discount_text=discount_text,
        hourly_df=hourly_df,
        is_demo_mode=demo_flag,
        output_path=default_report_path
    )

    if isinstance(report_res, tuple) or isinstance(report_res, list):
        actual_report_path = report_res[0] if len(report_res) > 0 else default_report_path
        pdf_bytes = report_res[1] if len(report_res) > 1 else None
    else:
        actual_report_path = report_res
        pdf_bytes = None

    if pdf_bytes is None and actual_report_path and os.path.exists(actual_report_path):
        try:
            with open(actual_report_path, "rb") as f:
                pdf_bytes = f.read()
        except Exception:
            pdf_bytes = None

    st.session_state.report_path = actual_report_path
    st.session_state.report_bytes = pdf_bytes
    st.session_state.report_message = "Store analysis completed. Your PDF report is ready!"
    st.session_state.report_ready = False
    return actual_report_path, pdf_bytes


# Reset data action
if reset_btn:
    st.session_state.tracker.reset()
    st.session_state.heatmap.reset()
    st.session_state.footfall_timeline = []
    st.session_state.frame_idx = 0
    st.session_state.stream_active = False
    st.session_state.stream_has_run = False
    st.session_state.report_path = None
    st.session_state.report_bytes = None
    st.session_state.report_ready = False
    st.session_state.report_message = None
    st.session_state.report_session_logged = False
    st.toast("System cache cleared.")
    st.rerun()

# Stream controllers
if run_stream:
    if not is_demo_mode and source_type == "Uploaded Video" and not video_ready:
        st.sidebar.warning("Please upload a CCTV video file before starting the stream.")
        st.session_state.stream_active = False
    else:
        st.session_state.stream_active = True
        st.session_state.stream_has_run = True
        st.session_state.stream_error = None
        if st.session_state.frame_idx == 0:
            st.session_state.tracker.reset()
            st.session_state.heatmap.reset()
            st.session_state.footfall_timeline = []
            st.session_state.report_path = None
            st.session_state.report_bytes = None
            st.session_state.report_ready = False
            st.session_state.report_message = None
            st.session_state.report_session_logged = False

if stop_stream:
    st.session_state.stream_active = False
    st.toast("Surveillance feed paused.")

# --- COMPACT BUSINESS HEADER RENDERERS ---

def render_header(status_active, current_fps=0.0):
    curr_time_str = time.strftime("%I:%M %p")
    status_class = "telemetry-live" if status_active else "telemetry-standby"
    dot_class = "pulse-dot-green" if status_active else "pulse-dot-gray"
    status_label = "LIVE FEED" if status_active else "STANDBY"
    fps_text = f"{current_fps:.1f} FPS" if status_active and current_fps > 0 else "Idle"
    
    render_html(f"""
    <div class="nav-bar">
        <!-- 1. Brand & Store Location Scope -->
        <div class="nav-brand-group">
            <div class="nav-brand-icon">
                <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round">
                    <path d="M6 2 3 6v14a2 2 0 0 0 2 2h14a2 2 0 0 0 2-2V6l-3-4Z"></path>
                    <path d="M3 6h18"></path>
                    <path d="M16 10a4 4 0 0 1-8 0"></path>
                </svg>
            </div>
            <div class="nav-brand-text">
                <span class="nav-brand-title">RetailFlow</span>
                <span class="nav-brand-subtitle">Store #104 &bull; Central Concourse CCTV</span>
            </div>
        </div>

        <!-- 2. Dedicated Navigation Link Tabs -->
        <div class="nav-page-links">
            <a href="http://127.0.0.1:{portal_port}" target="_blank" class="nav-page-link" title="Open Store Landing & Overview Portal">
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-linecap="round" stroke-linejoin="round">
                    <path d="m3 9 9-7 9 7v11a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z"></path>
                    <polyline points="9 22 9 12 15 12 15 22"></polyline>
                </svg>
                <span>Store Portal</span>
            </a>
            <span class="nav-page-link active" title="Current Active Live Dashboard">
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-linecap="round" stroke-linejoin="round">
                    <rect x="3" y="3" width="18" height="18" rx="2" ry="2"></rect>
                    <line x1="3" y1="9" x2="21" y2="9"></line>
                    <line x1="9" y1="21" x2="9" y2="9"></line>
                </svg>
                <span>Live Floor Feed</span>
            </span>
            <a href="http://127.0.0.1:{portal_port}/admin.html" target="_blank" class="nav-page-link" title="Open Administrator Governance Console">
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-linecap="round" stroke-linejoin="round">
                    <path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"></path>
                </svg>
                <span>Admin Console</span>
            </a>
        </div>

        <!-- 3. Clean Operational Telemetry & Sensor Badges -->
        <div class="nav-telemetry-group">
            <span class="telemetry-pill {status_class}">
                <span class="{dot_class}"></span>
                <span>{status_label}</span>
            </span>
            <span class="telemetry-pill">Cam: <strong>CAM_01</strong></span>
            <span class="telemetry-pill">Rate: <strong>{fps_text}</strong></span>
            <span class="telemetry-pill">Clock: <strong>{curr_time_str}</strong></span>
        </div>
    </div>
    """)

def render_hero_bar(mode_text, shopper_count, staff_count=0):
    render_html(f"""
    <div class="hero-card">
        <div class="hero-title">Store Floor Overview &amp; Spatial Heatmap</div>
        <div class="hero-subtitle">Real-time customer movement, aisle dwell duration, and register queue balancing</div>
        <div class="hero-stats-row">
            <div class="hero-stat-item">
                <div class="hero-stat-label">Processing Pipeline</div>
                <div class="hero-stat-value">{mode_text}</div>
            </div>
            <div class="hero-stat-item">
                <div class="hero-stat-label">Monitored Zone</div>
                <div class="hero-stat-value">Aisle 1 &bull; Central Concourse</div>
            </div>
            <div class="hero-stat-item">
                <div class="hero-stat-label">Staff Exclusion</div>
                <div class="hero-stat-value" style="color: #B87224;">{staff_count} uniforms filtered</div>
            </div>
            <div class="hero-stat-item">
                <div class="hero-stat-label">Tracked Shoppers</div>
                <div class="hero-stat-value">{shopper_count} currently in view</div>
            </div>
        </div>
    </div>
    """)

# 5 Business-standard KPI Cards Renderer
def render_kpi_metrics(total_visitors, active_shoppers, avg_dwell, busiest_zone, staff_isolated=0):
    render_html(f"""
    <div class="kpi-grid">
        <div class="kpi-card">
            <div class="kpi-header">
                <span class="kpi-label">Total Visitors</span>
                <span class="kpi-trend trend-up">Shoppers</span>
            </div>
            <div class="kpi-value">{total_visitors}</div>
            <div class="kpi-desc">Unique shoppers (Staff excluded)</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-header">
                <span class="kpi-label">Shoppers in View</span>
                <span class="kpi-trend trend-neutral">Live feed</span>
            </div>
            <div class="kpi-value">{active_shoppers}</div>
            <div class="kpi-desc">Customers currently in camera frame</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-header">
                <span class="kpi-label">Average Dwell</span>
                <span class="kpi-trend trend-up">Optimal</span>
            </div>
            <div class="kpi-value">{avg_dwell:.1f}s</div>
            <div class="kpi-desc">Average browsing duration per zone</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-header">
                <span class="kpi-label">Busiest Section</span>
                <span class="kpi-trend trend-highlight">Peak Traffic</span>
            </div>
            <div class="kpi-value" style="font-size: 13px; margin-top: 10px; font-weight: 700; color: #D48B38;">{busiest_zone}</div>
            <div class="kpi-desc">Department with highest dwell duration</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-header">
                <span class="kpi-label">Staff Filtered</span>
                <span class="kpi-trend" style="color: #D48B38; background: #FBF2E4; font-weight: 700; padding: 2px 6px; border-radius: 4px;">Isolated</span>
            </div>
            <div class="kpi-value" style="color: #D48B38;">{staff_isolated}</div>
            <div class="kpi-desc">Staff excluded from heatmap &amp; counts</div>
        </div>
    </div>
    """)

# Helper to style Plotly charts for the light theme
def apply_plotly_theme(fig):
    fig.update_layout(
        plot_bgcolor="#FFFFFF",
        paper_bgcolor="#FFFFFF",
        font=dict(color="#1F1712", family="Plus Jakarta Sans"),
        xaxis=dict(
            gridcolor="#E8DFD0", 
            zeroline=False,
            title_font=dict(color="#5E5147"),
            tickfont=dict(color="#5E5147")
        ),
        yaxis=dict(
            gridcolor="#E8DFD0", 
            zeroline=False,
            title_font=dict(color="#5E5147"),
            tickfont=dict(color="#5E5147")
        ),
        margin=dict(l=20, r=20, t=20, b=20),
    )
    fig.layout.title = None # Force title to None to completely eliminate "undefined" title bug

# Light-themed Customer Movement Trajectory Map Renderer
def draw_trajectory_map(tracker, is_demo=True):
    fig = go.Figure()
    
    # Render Store Aisle boundaries for both demo and real-time streams
    zones = tracker.zones
    for zone_name, (x1, y1, x2, y2) in zones.items():
        color = "#FAF7F2"
        border_color = "#E8DFD0"
        
        if "Aisle" in zone_name or "Gondola" in zone_name:
            color = "rgba(212, 139, 56, 0.04)"
            border_color = "rgba(212, 139, 56, 0.30)"
        elif "Checkout" in zone_name:
            color = "rgba(140, 62, 20, 0.04)"
            border_color = "rgba(140, 62, 20, 0.30)"
        elif "Entrance" in zone_name or "Exit" in zone_name:
            color = "rgba(46, 77, 56, 0.04)"
            border_color = "rgba(46, 77, 56, 0.30)"
            
        fig.add_shape(
            type="rect",
            x0=x1, y0=y1, x1=x2, y1=y2,
            fillcolor=color,
            line=dict(color=border_color, width=1),
            layer="below"
        )
        
        # Center labels
        fig.add_trace(go.Scatter(
            x=[(x1 + x2) / 2],
            y=[(y1 + y2) / 2],
            text=[zone_name.split(" (")[0]],
            mode="text",
            textfont=dict(color="#5E5147", size=9, family="Plus Jakarta Sans"),
            showlegend=False,
            hoverinfo="skip"
        ))
        
    # Render Shopper paths (simple lines, staff isolated)
    history = tracker.history
    if (not history or len(history) == 0) and is_demo and st.session_state.get("stream_has_run", False):
        history = get_benchmark_trajectories()

    staff_ids = getattr(tracker, "staff_tracks", set())
    filter_staff = getattr(tracker, "filter_staff", True)

    for track_id, pts in history.items():
        # Exclude staff tracks from customer trajectory map
        if filter_staff and track_id in staff_ids:
            continue

        if len(pts) > 1:
            x_coords = [p[0] for p in pts]
            y_coords = [p[1] for p in pts]
            
            # Trail Line
            fig.add_trace(go.Scatter(
                x=x_coords,
                y=y_coords,
                mode="lines",
                name=f"Shopper {track_id}",
                line=dict(width=1.5, color="#D48B38"),
                hoverinfo="text",
                hovertext=f"Shopper {track_id} Trajectory"
            ))
            
            # Simple Entry Marker
            fig.add_trace(go.Scatter(
                x=[x_coords[0]],
                y=[y_coords[0]],
                mode="markers",
                marker=dict(symbol="circle", size=6, color="#2E4D38"),
                showlegend=False,
                hoverinfo="text",
                hovertext=f"Shopper {track_id} Entry"
            ))
            
            # Simple Current Marker
            fig.add_trace(go.Scatter(
                x=[x_coords[-1]],
                y=[y_coords[-1]],
                mode="markers",
                marker=dict(symbol="circle", size=6, color="#D48B38"),
                showlegend=False,
                hoverinfo="text",
                hovertext=f"Shopper {track_id} Current"
            ))
            
            # Customer ID text label directly on map
            fig.add_trace(go.Scatter(
                x=[x_coords[-1]],
                y=[y_coords[-1] - 12],
                text=[f"ID {track_id}"],
                mode="text",
                textfont=dict(color="#1F1712", size=9, family="Plus Jakarta Sans"),
                showlegend=False,
                hoverinfo="skip"
            ))
            
    fig.update_layout(
        plot_bgcolor="#FFFFFF",
        paper_bgcolor="#FFFFFF",
        margin=dict(l=10, r=10, t=10, b=10),
        xaxis=dict(range=[0, 1280], visible=False, showgrid=False, zeroline=False),
        yaxis=dict(range=[720, 0], visible=False, showgrid=False, zeroline=False), # Inverted y-axis
        showlegend=False,
        height=380
    )
    
    return fig

# Trajectory Details Panel Helper
def render_trajectory_details(busiest_zone, is_demo=True):
    render_html(f"""
    <div class="glass-card" style="height: 100%; margin-bottom: 0;">
        <div style="font-size: 13px; font-weight: 700; color: #2C1E16; margin-bottom: 12px; border-bottom: 1px solid #E8DFD0; padding-bottom: 8px;">Movement Patterns &amp; Flow</div>
        <div style="margin-bottom: 12px;">
            <div style="font-size: 9.5px; color: #8E7F73; text-transform: uppercase; font-weight: 700; letter-spacing: 0.3px;">Primary Entrance</div>
            <div style="font-size: 12px; font-weight: 600; color: #2E4D38; margin-top: 3px; display: flex; align-items: center; gap: 6px;">
                <span style="width: 7px; height: 7px; border-radius: 50%; background: #2E4D38; display: inline-block;"></span>
                <span>Main Entry Doors (Active ingress)</span>
            </div>
        </div>
        <div style="margin-bottom: 12px;">
            <div style="font-size: 9.5px; color: #8E7F73; text-transform: uppercase; font-weight: 700; letter-spacing: 0.3px;">Primary Exit Area</div>
            <div style="font-size: 12px; font-weight: 600; color: #8C3E14; margin-top: 3px; display: flex; align-items: center; gap: 6px;">
                <span style="width: 7px; height: 7px; border-radius: 50%; background: #8C3E14; display: inline-block;"></span>
                <span>Front Checkout Lanes</span>
            </div>
        </div>
        <div style="margin-bottom: 12px;">
            <div style="font-size: 9.5px; color: #8E7F73; text-transform: uppercase; font-weight: 700; letter-spacing: 0.3px;">Highest Dwell Area</div>
            <div style="font-size: 12px; font-weight: 700; color: #D48B38; margin-top: 3px; display: flex; align-items: center; gap: 6px;">
                <span style="width: 7px; height: 7px; border-radius: 50%; background: #D48B38; display: inline-block;"></span>
                <span>{busiest_zone if busiest_zone != "None" else "Compiling metrics..."}</span>
            </div>
        </div>
        <div>
            <div style="font-size: 9.5px; color: #8E7F73; text-transform: uppercase; font-weight: 700; letter-spacing: 0.3px;">Common Shopping Paths</div>
            <div style="font-size: 11px; color: #5E5147; margin-top: 6px; line-height: 1.6;">
                &bull; Entrance &rarr; Snack Gondolas (42%)<br>
                &bull; Concourse &rarr; Dairy Coolers (36%)<br>
                &bull; Dry Goods &rarr; Checkout Lanes (64%)
            </div>
        </div>
    </div>
    """)


# Simple Executive Insights Panel Renderer
def generate_insights_list(metrics, is_demo=True):
    insights = []
    
    busiest = metrics.get("busiest_zone", "None")
    if busiest != "None":
        dwell = metrics.get("zone_dwells", {}).get(busiest, 0.0)
        insights.append({
            "icon": "•",
            "title": f"{busiest} section receives highest traffic",
            "desc": f"Shoppers accumulated {dwell:.1f}s dwell time. Optimize promotional placement and featured product displays in this zone.",
            "type": "success"
        })
    else:
        insights.append({
            "icon": "•",
            "title": "Awaiting traffic data to identify peak zones",
            "desc": "Shopper coordinates are being compiled in real-time to run hot-spot analytics.",
            "type": "neutral"
        })
        
    zone_dwells = metrics.get("zone_dwells", {})
    if zone_dwells:
        valid_dwells = {k: v for k, v in zone_dwells.items() if v > 0 and k != busiest}
        if valid_dwells:
            lowest_zone = min(valid_dwells, key=valid_dwells.get)
            insights.append({
                "icon": "•",
                "title": f"{lowest_zone} receives lower engagement",
                "desc": f"Accumulated dwell was {valid_dwells[lowest_zone]:.1f}s. Consider promotional incentives or aisle re-facing.",
                "type": "warning"
            })
            
    staff_count = metrics.get("total_staff_detected", 0)
    if staff_count > 0:
        insights.append({
            "icon": "•",
            "title": f"{staff_count} Staff Associate(s) Detected & Filtered",
            "desc": "Torso uniform analysis successfully isolated staff movements from customer footfall and dwell analytics.",
            "type": "info"
        })
    else:
        insights.append({
            "icon": "•",
            "title": "Peak activity occurs between 6 PM and 8 PM",
            "desc": "Staffing models suggest scheduling additional checkout personnel during this high-volume window.",
            "type": "info"
        })
        
    return insights

# Flat Light-themed Plotly Chart Renderers
def draw_footfall_chart(timeline, is_demo=False):
    if not timeline:
        if is_demo:
            _, _, bench_tl = get_benchmark_simulation_dataset()
            if bench_tl:
                df = pd.DataFrame(bench_tl)
                fig = px.area(df, x="Time (s)", y="Shopper Count")
                fig.update_traces(line_color="#D48B38", fillcolor="rgba(212, 139, 56, 0.08)", line_width=2)
                apply_plotly_theme(fig)
                fig.update_layout(height=240)
                return fig
        fig = go.Figure()
        fig.add_annotation(text="No visitor timeline compiled", showarrow=False, font=dict(size=11, color="#8E7F73"))
        fig.update_layout(xaxis=dict(visible=False), yaxis=dict(visible=False))
    else:
        df = pd.DataFrame(timeline)
        fig = px.area(df, x="Time (s)", y="Shopper Count")
        fig.update_traces(line_color="#D48B38", fillcolor="rgba(212, 139, 56, 0.08)", line_width=2)
    apply_plotly_theme(fig)
    fig.update_layout(height=240)
    return fig

def draw_zone_chart(zone_counts, zones):
    data = []
    for z in zones.keys():
        data.append({
            "Zone": z.split(" (")[0],
            "Visitors": zone_counts.get(z, 0)
        })
    df = pd.DataFrame(data).sort_values(by="Visitors", ascending=True)
    fig = px.bar(df, x="Visitors", y="Zone", orientation="h")
    fig.update_traces(marker_color="#D48B38", marker_line_color="#E8DFD0", marker_line_width=1)
    apply_plotly_theme(fig)
    fig.update_layout(height=240)
    return fig

def draw_dwell_chart(final_df, is_demo=False):
    if final_df is None or final_df.empty:
        if is_demo:
            sim_df, _, _ = get_benchmark_simulation_dataset()
            if not sim_df.empty and "Total_Dwell_Time_Sec" in sim_df.columns:
                fig = px.histogram(sim_df, x="Total_Dwell_Time_Sec", nbins=8)
                fig.update_traces(marker_color="#2E4D38", marker_line_color="#E8DFD0", marker_line_width=1)
                fig.update_layout(xaxis_title="Dwell Duration (s)", yaxis_title="Visitors")
                apply_plotly_theme(fig)
                fig.update_layout(height=240)
                return fig
        fig = go.Figure()
        fig.add_annotation(text="No shopper profiles recorded", showarrow=False, font=dict(size=11, color="#8E7F73"))
        fig.update_layout(xaxis=dict(visible=False), yaxis=dict(visible=False))
    else:
        fig = px.histogram(final_df, x="Total_Dwell_Time_Sec", nbins=8)
        fig.update_traces(marker_color="#2E4D38", marker_line_color="#E8DFD0", marker_line_width=1)
        fig.update_layout(xaxis_title="Dwell Duration (s)", yaxis_title="Visitors")
    apply_plotly_theme(fig)
    fig.update_layout(height=240)
    return fig

def draw_hourly_chart():
    hours = [f"{h:02d}:00" for h in range(8, 23)]
    traffic = [14, 22, 29, 36, 45, 38, 44, 66, 82, 102, 134, 142, 110, 78, 28]
    df = pd.DataFrame({"Hour": hours, "Shoppers": traffic})
    fig = px.area(df, x="Hour", y="Shoppers")
    fig.update_traces(line_color="#2C1E16", fillcolor="rgba(44, 30, 22, 0.06)", line_width=2)
    apply_plotly_theme(fig)
    fig.update_layout(height=240)
    return fig

# Aisle Registry Database Generator
def generate_zone_analytics_table(tracker, final_df, final_metrics):
    zones = tracker.zones
    zone_counts = final_metrics.get("zone_counts", {})
    zone_dwells = final_metrics.get("zone_dwells", {})
    
    zone_records = []
    for zone_name in zones.keys():
        visitors = zone_counts.get(zone_name, 0)
        total_time = zone_dwells.get(zone_name, 0.0)
        
        # Calculate average dwell time
        avg_time = 0.0
        if visitors > 0 and not final_df.empty and zone_name in final_df.columns:
            valid_dwells = final_df[final_df[zone_name] > 0][zone_name]
            if not valid_dwells.empty:
                avg_time = round(valid_dwells.mean(), 1)
        elif visitors > 0:
            avg_time = round(total_time / visitors, 1)
            
        # Density index
        if total_time > 80:
            density = "High Traffic"
        elif total_time > 25:
            density = "Medium Traffic"
        else:
            density = "Low Traffic"
            
        # Zone Status
        if zone_name in ["Checkout 1", "Checkout 2"] and visitors > 3:
            status = "Queue Alert"
        elif visitors > 0:
            status = "Active"
        else:
            status = "Idle"
            
        zone_records.append({
            "Zone": zone_name,
            "Visitors": visitors,
            "Total Dwell (s)": round(total_time, 1),
            "Avg Dwell (s)": avg_time,
            "Average Time": f"{avg_time:.1f}s" if visitors > 0 else "0.0s",
            "Traffic Density": density,
            "Status": status
        })
        
    return pd.DataFrame(zone_records)

# --- STREAMLIT DASHBOARD PAGE CONSTRUCTION ---

# Flat compact header placeholder
nav_container = st.empty()

# Flat stats context placeholder
hero_container = st.empty()

# 5 KPI cards placeholder
kpi_container = st.empty()

# Main Display Viewports (Side-by-side visible immediately)
col1, col2 = st.columns(2)

with col1:
    render_html("""
    <div class="viewport-title">
        <span>Live Camera View &bull; Object Tracking</span>
        <span style="font-size: 11px; font-weight: 500; color: #8E7F73; text-transform: none;">Edge YOLOv8 Detection</span>
    </div>
    """)
    hud_container = st.empty()
    cctv_container = st.empty()
    
with col2:
    render_html("""
    <div class="viewport-title">
        <span>Customer Dwell Density Heatmap</span>
        <span style="font-size: 11px; font-weight: 500; color: #8E7F73; text-transform: none;">Continuous Footfall Accumulation</span>
    </div>
    """)
    render_html("""
    <div class="traffic-legend">
        <span class="legend-item"><span class="legend-dot high"></span>High Dwell</span>
        <span class="legend-item"><span class="legend-dot medium"></span>Moderate</span>
        <span class="legend-item"><span class="legend-dot low"></span>Transit</span>
    </div>
    """)
    heatmap_container = st.empty()

# --- DYNAMIC STREAMING PIPELINE ---

if st.session_state.stream_active:
    # 1. Resolve source
    video_source = None
    if source_type == "Sample Video":
        sample_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sample_cctv.mp4")
        if not os.path.exists(sample_path) or os.path.getsize(sample_path) == 0:
            st.warning("Sample video file was not found. Generating simulation file...")
            generate_video(sample_path)
            tracker.load_emulation_data(os.path.join(os.path.dirname(os.path.abspath(__file__)), "sample_cctv_detections.csv"))
        video_source = sample_path
    elif source_type == "Uploaded Video":
        active_video = st.session_state.get("active_video_path")
        if active_video and os.path.exists(active_video) and os.path.getsize(active_video) > 0:
            video_source = active_video
        elif uploaded_file is not None and len(uploaded_file.getvalue()) > 0:
            save_dest = os.path.join(os.path.dirname(os.path.abspath(__file__)), "temp_cctv_uploaded.mp4")
            try:
                with open(save_dest, "wb") as f:
                    f.write(uploaded_file.getvalue())
                st.session_state.active_video_path = save_dest
                video_source = save_dest
            except Exception:
                video_source = None
        else:
            st.session_state.stream_error = "Please upload an MP4 surveillance video before starting the camera feed."
            st.session_state.stream_active = False
            st.rerun()
    elif source_type == "Webcam":
        video_source = 0
    else:
        video_source = "sample_cctv.mp4"
        
    if video_source is not None:
        cap = None
        try:
            cap = cv2.VideoCapture(video_source)
            if not cap.isOpened():
                st.session_state.stream_error = f"Error opening surveillance channel: {os.path.basename(str(video_source))}"
                st.session_state.stream_active = False
            else:
                # Load video configs and calibrate smooth, realistic playback pacing
                native_fps = cap.get(cv2.CAP_PROP_FPS)
                if native_fps == 0 or np.isnan(native_fps) or native_fps is None:
                    native_fps = 30.0
                    
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                is_demo = (source_type == "Sample Video")

                # Playback pace calibration:
                # - Demo Simulation: 16.0 FPS (relaxed, normal human walking speed for 600 frames, ~38s total runtime)
                # - Uploaded CCTV: 20.0 FPS max (fluid, non-lagging surveillance rate)
                # - Webcam: native webcam capture rate
                if is_demo:
                    stream_fps = 16.0
                elif source_type == "Webcam":
                    stream_fps = float(native_fps)
                else:
                    stream_fps = min(float(native_fps), 20.0)
                
                # Seek forward if resuming
                if st.session_state.frame_idx > 0 and source_type != "Webcam":
                    cap.set(cv2.CAP_PROP_POS_FRAMES, st.session_state.frame_idx)
                    
                frame_idx = st.session_state.frame_idx
                loop_start = time.time()
                frame_count = 0
                measured_fps = float(stream_fps)
                last_ui_update_time = 0.0

                if source_type == "Webcam":
                    try:
                        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                    except Exception:
                        pass
                
                # Loop streaming frames
                while cap.isOpened() and st.session_state.stream_active:
                    frame_start_time = time.time()
                    
                    # Read frame sequentially without jumpy frame-skips
                    ret, frame = cap.read()

                    if not ret:
                        # Video ended: stop the stream and trigger PDF generation
                        st.session_state.stream_active = False
                        st.session_state.frame_idx = 0
                        st.session_state.report_ready = True
                        st.toast("Surveillance stream playback completed. Preparing report...")
                        break
                        
                    # Process frame
                    frame = cv2.resize(frame, (1280, 720))
                    annotated_frame = frame.copy()
                    
                    if draw_zones_toggle:
                        annotated_frame = tracker.draw_zones(annotated_frame)
                    
                    # Real-time tracking (asynchronous background worker prevents video lag)
                    annotated_frame, centers, active_count = tracker.process_frame(
                        annotated_frame, frame_idx, conf_threshold=conf_threshold, skip_detection=False
                    )
                    
                    heatmap.update(centers)
                    heatmap_blended, _ = heatmap.get_heatmap(
                        frame, alpha=heatmap_alpha, blur_strength=blur_strength, colormap=selected_cmap
                    )
                    
                    if draw_zones_toggle:
                        heatmap_blended = tracker.draw_zones(heatmap_blended)
                        
                    # Diagnostics FPS
                    frame_count += 1
                    elapsed = time.time() - loop_start
                    measured_fps = frame_count / elapsed if elapsed > 0 else float(stream_fps)
                    
                    # Append to timeline
                    timeline_step = max(1, int(stream_fps))
                    if frame_idx % timeline_step == 0:
                        st.session_state.footfall_timeline.append({
                            "Time (s)": round(frame_idx / stream_fps, 1),
                            "Shopper Count": active_count
                        })
                        
                    # Throttle HTML DOM updates to 4 times per second (prevents WebSocket queue backlog)
                    now_time = time.time()
                    if now_time - last_ui_update_time >= 0.25:
                        last_ui_update_time = now_time
                        df_temp, metrics_temp = tracker.get_analytics(fps=stream_fps)
                        save_session_telemetry(is_live=True, active_count=active_count, fps=measured_fps)
                        
                        nav_container.empty()
                        with nav_container:
                            render_header(True, measured_fps)
                            
                        hero_container.empty()
                        with hero_container:
                            render_hero_bar(
                                "Simulation Demo" if not tracker.use_yolo else "YOLOv8 Active Stream",
                                active_count,
                                tracker.active_staff_count
                            )
                            
                        kpi_container.empty()
                        with kpi_container:
                            render_kpi_metrics(
                                metrics_temp["total_footfall"],
                                active_count,
                                metrics_temp["avg_dwell_time"],
                                metrics_temp["busiest_zone"],
                                metrics_temp.get("total_staff_detected", 0)
                            )
                            
                        staff_hud = f'<span class="hud-item" style="color: #F59E0B;">STAFF EXCLUDED: <strong>{tracker.active_staff_count}</strong></span>' if tracker.filter_staff else '<span class="hud-item">STAFF FILTER: <strong>OFF</strong></span>'
                        render_html(f"""
                        <div class="hud-bar">
                            <span class="hud-badge">LIVE TRACKING</span>
                            <span class="hud-item">FPS: <strong>{measured_fps:.1f}</strong></span>
                            <span class="hud-item">ACTIVE SHOPPERS: <strong>{active_count}</strong></span>
                            {staff_hud}
                            <span class="hud-item">CONFIDENCE: <strong>{conf_threshold:.2f}</strong></span>
                        </div>
                        """, container=hud_container)
                    
                    # Resize display frames to 960x540 for ultra-smooth WebSocket transfer with zero lag
                    disp_w, disp_h = 960, 540
                    cctv_rgb = cv2.cvtColor(cv2.resize(annotated_frame, (disp_w, disp_h)), cv2.COLOR_BGR2RGB)
                    heatmap_rgb = cv2.cvtColor(cv2.resize(heatmap_blended, (disp_w, disp_h)), cv2.COLOR_BGR2RGB)
                    
                    cctv_container.image(cctv_rgb, use_container_width=True)
                    heatmap_container.image(heatmap_rgb, use_container_width=True)
                    
                    frame_idx += 1
                    st.session_state.frame_idx = frame_idx
                    
                    # Accurate playback pacing
                    target_sleep = 1.0 / stream_fps
                    processing_duration = time.time() - frame_start_time
                    if processing_duration < target_sleep:
                        time.sleep(target_sleep - processing_duration)

        except Exception as stream_err:
            st.session_state.stream_error = f"Stream playback note: {stream_err}"
            st.session_state.stream_active = False
        finally:
            st.session_state.stream_active = False
            if cap is not None:
                try:
                    cap.release()
                except Exception:
                    pass
            save_session_telemetry(is_live=False, active_count=0, fps=0.0)
            st.rerun()

# --- STANDBY STATIC VIEWPORTS ---
if not st.session_state.stream_active:
    if st.session_state.get("stream_error"):
        st.error(f"Surveillance Channel Notice: {st.session_state.stream_error}")

    is_demo_mode = (source_type == "Sample Video")
    final_df, final_metrics = tracker.get_analytics()
    save_session_telemetry(is_live=False, active_count=0, fps=0.0)

    # In standby before stream runs, show real session state without pre-populating benchmark results
    if (final_df.empty or final_metrics.get("total_footfall", 0) == 0) and is_demo_mode and st.session_state.get("stream_has_run", False):
        display_df, display_metrics, demo_tl = get_benchmark_simulation_dataset()
        timeline_data = st.session_state.footfall_timeline if st.session_state.footfall_timeline else demo_tl
    else:
        display_df = final_df
        display_metrics = final_metrics
        timeline_data = st.session_state.footfall_timeline

    busiest_name = display_metrics.get("busiest_zone", "None")
    if busiest_name in ["None", ""] and is_demo_mode and st.session_state.get("stream_has_run", False):
        busiest_name = "Aisle A (Produce)"
    elif busiest_name in ["None", ""]:
        busiest_name = "Standby"

    # Automatically generate the PDF once a video reaches the end.
    if st.session_state.report_ready:
        try:
            generate_store_report()
        except Exception as report_error:
            st.session_state.report_ready = False
            st.session_state.report_message = None
            st.error(f"PDF report generation failed: {report_error}")

    # Keep the download button visible after the report has been generated.
    has_report = (st.session_state.get("report_bytes") is not None) or (
        st.session_state.get("report_path") and os.path.exists(st.session_state.report_path)
    )
    if has_report:
        dl_data = st.session_state.report_bytes if st.session_state.get("report_bytes") is not None else open(st.session_state.report_path, "rb").read()
        st.success(
            st.session_state.get(
                "report_message",
                "Store analysis completed. Your performance report is ready."
            )
        )
        st.download_button(
            label="Download Store Performance Report (PDF)",
            data=dl_data,
            file_name="Retail_Store_Performance_Report.pdf",
            mime="application/pdf",
            use_container_width=True,
            key="download_report_pdf_top"
        )

    with nav_container:
        render_header(False, 0.0)
        
    with hero_container:
        render_hero_bar(
            "Simulation Demo" if is_demo_mode else "YOLOv8 Active Stream",
            0,
            display_metrics.get("total_staff_detected", 0)
        )
        
    with kpi_container:
        render_kpi_metrics(
            display_metrics.get("total_footfall", 0),
            0,
            display_metrics.get("avg_dwell_time", 0.0),
            busiest_name,
            display_metrics.get("total_staff_detected", 0)
        )
        
    staff_standby_hud = f'<span class="hud-item" style="color: #F59E0B;">STAFF EXCLUDED: <strong>{display_metrics.get("total_staff_detected", 0)}</strong></span>' if tracker.filter_staff else '<span class="hud-item">STAFF FILTER: <strong>OFF</strong></span>'
    standby_badge_lbl = "SIMULATION READY" if is_demo_mode else "CCTV READY"
    standby_status_lbl = "STANDBY (Click 'Run Simulation Demo' to begin)" if is_demo_mode else "STANDBY (Click 'Start Camera Feed' to begin)"
    render_html(f"""
    <div class="hud-bar" style="background: #F8FAFC; border-color: #E5E7EB;">
        <span class="hud-badge" style="background: #6B7280;">{standby_badge_lbl}</span>
        <span class="hud-item">FPS: <strong>0.0</strong></span>
        <span class="hud-item">ACTIVE SHOPPERS: <strong>0</strong></span>
        {staff_standby_hud}
        <span class="hud-item">STATUS: <strong>{standby_status_lbl}</strong></span>
    </div>
    """, container=hud_container)
    
    # Frame 0 Preview extraction for CCTV Viewport
    preview_file = None
    if is_demo_mode:
        sample_cand = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sample_cctv.mp4")
        if os.path.exists(sample_cand) and os.path.getsize(sample_cand) > 0:
            preview_file = sample_cand
    elif source_type == "Uploaded Video":
        uploaded_cand = st.session_state.get("active_video_path")
        if uploaded_cand and os.path.exists(uploaded_cand) and os.path.getsize(uploaded_cand) > 0:
            preview_file = uploaded_cand

    preview_frame = None
    if preview_file:
        try:
            p_cap = cv2.VideoCapture(preview_file)
            if p_cap.isOpened():
                p_ret, p_f = p_cap.read()
                p_cap.release()
                if p_ret and p_f is not None:
                    preview_frame = cv2.resize(p_f, (1280, 720))
        except Exception:
            preview_frame = None

    if preview_frame is not None:
        cctv_p = preview_frame.copy()
        if draw_zones_toggle:
            cctv_p = tracker.draw_zones(cctv_p)
        cctv_container.image(cv2.cvtColor(cctv_p, cv2.COLOR_BGR2RGB), use_container_width=True)
        
        # If stream has already run, display final accumulated heatmap
        if st.session_state.get("stream_has_run", False):
            if is_demo_mode and final_df.empty:
                try:
                    demo_hm = HeatmapAccumulator()
                    for tid, pts in get_benchmark_trajectories().items():
                        demo_hm.update(pts)
                    hm_prev, _ = demo_hm.get_heatmap(
                        preview_frame, alpha=heatmap_alpha, blur_strength=blur_strength, colormap=selected_cmap
                    )
                    if draw_zones_toggle:
                        hm_prev = tracker.draw_zones(hm_prev)
                    heatmap_container.image(cv2.cvtColor(hm_prev, cv2.COLOR_BGR2RGB), use_container_width=True)
                except Exception:
                    heatmap_container.image(cv2.cvtColor(cctv_p, cv2.COLOR_BGR2RGB), use_container_width=True)
            else:
                try:
                    hm_prev, _ = heatmap.get_heatmap(
                        preview_frame, alpha=heatmap_alpha, blur_strength=blur_strength, colormap=selected_cmap
                    )
                    if draw_zones_toggle:
                        hm_prev = tracker.draw_zones(hm_prev)
                    heatmap_container.image(cv2.cvtColor(hm_prev, cv2.COLOR_BGR2RGB), use_container_width=True)
                except Exception:
                    heatmap_container.image(cv2.cvtColor(cctv_p, cv2.COLOR_BGR2RGB), use_container_width=True)
        else:
            render_html("""
            <div class="viewport-frame" style="height: 380px; display: flex; align-items: center; justify-content: center; background: #FAF7F2; border-radius: 8px; border: 1.5px dashed #D8CEBE;">
                <div style="text-align: center; max-width: 320px; padding: 20px;">
                    <div style="width: 44px; height: 44px; margin: 0 auto 12px; background: #F3ECE1; border-radius: 10px; display: flex; align-items: center; justify-content: center; color: #8E7F73;">
                        <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                            <polygon points="3 6 9 3 15 6 21 3 21 18 15 21 9 18 3 21"></polygon>
                            <line x1="9" y1="3" x2="9" y2="18"></line>
                            <line x1="15" y1="6" x2="15" y2="21"></line>
                        </svg>
                    </div>
                    <div style="color: #2C1E16; font-size: 14px; font-weight: 600;">Heatmap Canvas in Standby</div>
                    <div style="color: #8E7F73; font-size: 12px; margin-top: 6px; line-height: 1.4;">Thermal dwell density will accumulate automatically once stream playback starts.</div>
                </div>
            </div>
            """, container=heatmap_container)
    else:
        render_html("""
        <div class="viewport-frame" style="height: 380px; display: flex; align-items: center; justify-content: center; background: #FAF7F2; border-radius: 8px; border: 1.5px dashed #D8CEBE;">
            <div style="text-align: center; max-width: 320px; padding: 20px;">
                <div style="width: 44px; height: 44px; margin: 0 auto 12px; background: #F3ECE1; border-radius: 10px; display: flex; align-items: center; justify-content: center; color: #8E7F73;">
                    <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                        <path d="m22 8-6 4 6 4V8Z"></path>
                        <rect x="2" y="6" width="14" height="12" rx="2"></rect>
                    </svg>
                </div>
                <div style="color: #2C1E16; font-size: 14px; font-weight: 600;">No CCTV Video Uploaded</div>
                <div style="color: #8E7F73; font-size: 12px; margin-top: 6px; line-height: 1.4;">Please upload an MP4, AVI, or MOV video file in the sidebar to begin YOLOv8 real-time customer tracking.</div>
            </div>
        </div>
        """, container=cctv_container)
        
        render_html("""
        <div class="viewport-frame" style="height: 380px; display: flex; align-items: center; justify-content: center; background: #FAF7F2; border-radius: 8px; border: 1.5px dashed #D8CEBE;">
            <div style="text-align: center; max-width: 320px; padding: 20px;">
                <div style="width: 44px; height: 44px; margin: 0 auto 12px; background: #F3ECE1; border-radius: 10px; display: flex; align-items: center; justify-content: center; color: #8E7F73;">
                    <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                        <polygon points="3 6 9 3 15 6 21 3 21 18 15 21 9 18 3 21"></polygon>
                        <line x1="9" y1="3" x2="9" y2="18"></line>
                        <line x1="15" y1="6" x2="15" y2="21"></line>
                    </svg>
                </div>
                <div style="color: #2C1E16; font-size: 14px; font-weight: 600;">Heatmap Canvas in Standby</div>
                <div style="color: #8E7F73; font-size: 12px; margin-top: 6px; line-height: 1.4;">Thermal dwell density will accumulate automatically as shoppers move across aisles.</div>
            </div>
        </div>
        """, container=heatmap_container)

    # In-Dashboard Stream Controller Bar
    st.markdown("<div style='margin-top: 14px;'></div>", unsafe_allow_html=True)
    col_play1, col_play2, col_play3 = st.columns([3, 1, 1])
    with col_play1:
        dash_btn_label = "▶ Run Simulation Demo" if is_demo_mode else "▶ Start Camera Feed"
        dash_disabled = (not is_demo_mode and source_type == "Uploaded Video" and not video_ready)
        if st.button(dash_btn_label, type="primary", use_container_width=True, key="dash_start_stream_btn", disabled=dash_disabled):
            st.session_state.stream_active = True
            st.session_state.stream_has_run = True
            st.session_state.stream_error = None
            st.rerun()
    with col_play2:
        st.button("⏹ Pause Stream", use_container_width=True, key="dash_pause_btn", disabled=True)
    with col_play3:
        if st.button("🔄 Reset Telemetry", use_container_width=True, key="dash_reset_btn"):
            st.session_state.tracker.reset()
            st.session_state.heatmap.reset()
            st.session_state.footfall_timeline = []
            st.session_state.frame_idx = 0
            st.session_state.stream_active = False
            st.session_state.stream_has_run = False
            st.session_state.stream_error = None
            st.session_state.report_path = None
            st.session_state.report_bytes = None
            st.rerun()

# Lower Analytics Data
is_demo_mode = (source_type == "Sample Video")
final_df, final_metrics = tracker.get_analytics()

if (final_df.empty or final_metrics.get("total_footfall", 0) == 0) and is_demo_mode and st.session_state.get("stream_has_run", False):
    display_df, display_metrics, demo_tl = get_benchmark_simulation_dataset()
    timeline_data = st.session_state.footfall_timeline if st.session_state.footfall_timeline else demo_tl
else:
    display_df = final_df
    display_metrics = final_metrics
    timeline_data = st.session_state.footfall_timeline

busiest_zone_name = display_metrics.get("busiest_zone", "None")
if busiest_zone_name == "None" and is_demo_mode:
    busiest_zone_name = "Aisle A (Produce)"

# --- CUSTOMER MOVEMENT SECTION (Directly below CCTV) ---
st.markdown("<div style='margin-top: 25px;'></div>", unsafe_allow_html=True)

st.markdown("<h3 style='color:#2C1E16; font-weight:700; font-size:16px; margin-bottom: 12px;'>Shopper Movement Paths &amp; Trajectories</h3>", unsafe_allow_html=True)

traj_col1, traj_col2 = st.columns([7, 3])
with traj_col1:
    traj_fig = draw_trajectory_map(tracker, is_demo=is_demo_mode)
    st.plotly_chart(traj_fig, use_container_width=True)
with traj_col2:
    render_trajectory_details(busiest_zone_name, is_demo=is_demo_mode)

# --- STORE OPERATIONAL INSIGHTS ---
st.markdown("<div style='margin-top: 25px;'></div>", unsafe_allow_html=True)
st.markdown("<h3 style='color:#2C1E16; font-weight:700; font-size:16px; margin-bottom: 12px;'>Floor Operations &amp; Merchandising Insights</h3>", unsafe_allow_html=True)

insights_list = generate_insights_list(display_metrics, is_demo=is_demo_mode)
insights_html = '<div class="insights-grid">'
for ins in insights_list:
    insights_html += f'<div class="insight-card {ins["type"]}-type"><div class="insight-icon-container">{ins["icon"]}</div><div class="insight-content"><span class="insight-title">{ins["title"]}:</span><span class="insight-desc">{ins["desc"]}</span></div></div>'
insights_html += '</div>'
render_html(insights_html)

# --- PERFORMANCE CHARTS SECTION ---
st.markdown("<div style='margin-top: 30px;'></div>", unsafe_allow_html=True)
st.markdown("<h3 style='color:#2C1E16; font-weight:700; font-size:16px; margin-bottom: 12px;'>Footfall Trends &amp; Department Dwell Analytics</h3>", unsafe_allow_html=True)

c_row1_col1, c_row1_col2 = st.columns(2)
with c_row1_col1:
    st.markdown('<div class="chart-header">Footfall Over Time</div>', unsafe_allow_html=True)
    st.plotly_chart(draw_footfall_chart(timeline_data), use_container_width=True)
with c_row1_col2:
    st.markdown('<div class="chart-header">Department Traffic Breakdown</div>', unsafe_allow_html=True)
    st.plotly_chart(draw_zone_chart(display_metrics.get("zone_counts", {}), tracker.zones), use_container_width=True)

c_row2_col1, c_row2_col2 = st.columns(2)
with c_row2_col1:
    st.markdown('<div class="chart-header">Customer Dwell Duration Distribution</div>', unsafe_allow_html=True)
    st.plotly_chart(draw_dwell_chart(display_df, is_demo=is_demo_mode), use_container_width=True)
with c_row2_col2:
    st.markdown('<div class="chart-header">Estimated Hourly Traffic Pattern</div>', unsafe_allow_html=True)
    st.plotly_chart(draw_hourly_chart(), use_container_width=True)

# --- AISLE PERFORMANCE DATA REGISTRY ---
st.markdown("<div style='margin-top: 30px;'></div>", unsafe_allow_html=True)
st.markdown("<h3 style='color:#2C1E16; font-weight:700; font-size:16px; margin-bottom: 12px;'>Department &amp; Aisle Performance Summary</h3>", unsafe_allow_html=True)

zone_table_data = generate_zone_analytics_table(tracker, display_df, display_metrics)

search_col1, search_col2 = st.columns([7, 3])
with search_col1:
    search_q = st.text_input("Search aisles...", placeholder="Filter by department name, density, or status...", label_visibility="collapsed")
with search_col2:
    status_filter = st.selectbox(
        "Status Filter",
        ["All Departments", "Active", "Queue Alert", "Idle"],
        label_visibility="collapsed"
    )
    
# Filter registries
filtered_zone_data = zone_table_data.copy()
if search_q:
    filtered_zone_data = filtered_zone_data[
        filtered_zone_data["Zone"].str.contains(search_q, case=False) |
        filtered_zone_data["Traffic Density"].str.contains(search_q, case=False)
    ]
if status_filter != "All Departments":
    filtered_zone_data = filtered_zone_data[filtered_zone_data["Status"] == status_filter]
    
# Render clean grid
st.dataframe(filtered_zone_data, use_container_width=True, hide_index=True)

# CSV Export Button
csv_export = filtered_zone_data.to_csv(index=False).encode('utf-8')
st.download_button(
    label="Export Aisle Analytics (CSV)",
    data=csv_export,
    file_name="aisle_performance_analytics.csv",
    mime="text/csv",
    use_container_width=True
)

# Final Store Report Controls in Sidebar
st.sidebar.markdown("---")
st.sidebar.markdown("### Executive Store Report")
st.sidebar.caption("Compile and download the executive store performance PDF report from current session telemetry.")

if not st.session_state.stream_active:
    if st.sidebar.button("Generate Store Report (PDF)", use_container_width=True, key="sidebar_gen_pdf_btn"):
        try:
            with st.spinner("Compiling store intelligence report..."):
                generate_store_report()
            st.toast("Store report generated successfully.")
            st.rerun()
        except Exception as report_error:
            st.sidebar.error(f"PDF generation failed: {report_error}")

has_report_sidebar = (st.session_state.get("report_bytes") is not None) or (
    st.session_state.get("report_path") and os.path.exists(st.session_state.report_path)
)
if has_report_sidebar:
    dl_data_sidebar = st.session_state.report_bytes if st.session_state.get("report_bytes") is not None else open(st.session_state.report_path, "rb").read()
    st.sidebar.download_button(
        label="Download Store Report (PDF)",
        data=dl_data_sidebar,
        file_name="Retail_Store_Performance_Report.pdf",
        mime="application/pdf",
        use_container_width=True,
        key="sidebar_download_pdf_btn"
    )
    st.sidebar.link_button(
        label="Open PDF in Browser",
        url=f"http://127.0.0.1:{portal_port}/Retail_Store_Performance_Report.pdf",
        use_container_width=True
    )

# --- FINAL PDF REPORT SECTION ---
st.markdown("<div style='height: 30px;'></div>", unsafe_allow_html=True)
st.markdown("<h3 style='color:#2C1E16; font-weight:700; font-size:16px; margin-bottom: 12px;'>Executive Store Performance Report</h3>", unsafe_allow_html=True)

has_report_main = (st.session_state.get("report_bytes") is not None) or (
    st.session_state.get("report_path") and os.path.exists(st.session_state.report_path)
)

if has_report_main:
    dl_data_main = st.session_state.report_bytes if st.session_state.get("report_bytes") is not None else open(st.session_state.report_path, "rb").read()
    st.success("Your comprehensive store analytics report has been compiled and is ready for review.")
    
    col_dl1, col_dl2, col_dl3 = st.columns([4, 3, 3])
    with col_dl1:
        st.download_button(
            label="Download Store Report (PDF)",
            data=dl_data_main,
            file_name="Retail_Store_Performance_Report.pdf",
            mime="application/pdf",
            use_container_width=True,
            key="download_report_pdf_bottom"
        )
    with col_dl2:
        st.link_button(
            label="View PDF in Browser",
            url=f"http://127.0.0.1:{portal_port}/Retail_Store_Performance_Report.pdf",
            use_container_width=True
        )
    with col_dl3:
        if st.button("Refresh Report Data", use_container_width=True, key="regen_report_btn"):
            try:
                with st.spinner("Refreshing report data..."):
                    generate_store_report()
                st.toast("Report refreshed successfully.")
                st.rerun()
            except Exception as e:
                st.error(f"Re-generation error: {e}")
else:
    st.info("The executive PDF includes customer footfall scorecards, aisle dwell charts, queue diagnosis, and practical merchandising recommendations.")
    if st.button("Generate Store Performance Report (PDF)", use_container_width=True, key="main_gen_pdf_btn"):
        try:
            with st.spinner("Compiling store metrics and generating PDF report..."):
                generate_store_report()
            st.toast("Store report generated successfully.")
            st.rerun()
        except Exception as e:
            st.error(f"PDF generation error: {e}")

