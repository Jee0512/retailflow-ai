import os
import sys
import json
import socket
import http.server
import socketserver
import threading
import time

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

class PortalHandler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=ROOT_DIR, **kwargs)

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

        # Route aliases
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

        # PDF Report Generation / Download
        if clean_path in ["/Retail_Store_Performance_Report.pdf", "/generate_report", "/generate-report"]:
            pdf_file = os.path.join(ROOT_DIR, "Retail_Store_Performance_Report.pdf")
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
                # 1. Check if Streamlit is alive on port 8501
                streamlit_alive = False
                try:
                    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                        s.settimeout(0.2)
                        streamlit_alive = (s.connect_ex(("127.0.0.1", 8501)) == 0)
                except Exception:
                    streamlit_alive = False

                telemetry_file = os.path.join(ROOT_DIR, "session_telemetry.json")
                telemetry_data = None
                if os.path.exists(telemetry_file):
                    try:
                        with open(telemetry_file, "r", encoding="utf-8") as tf:
                            telemetry_data = json.load(tf)
                    except Exception:
                        telemetry_data = None

                now = time.time()
                is_live = False
                if telemetry_data and streamlit_alive:
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
                    # System is in Standby / Benchmark Preset mode
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
                    "streamlit_alive": streamlit_alive,
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

        # Admin Users List
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

        # Admin Telemetry Stats
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

        # Admin Audit Logs
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

def start_portal_server(port=8000):
    """Starts the portal HTTP server on port 8000 in a daemon thread."""
    socketserver.TCPServer.allow_reuse_address = True
    for host in ["", "0.0.0.0", "127.0.0.1"]:
        try:
            httpd = socketserver.TCPServer((host, port), PortalHandler)
            t = threading.Thread(target=httpd.serve_forever, daemon=True)
            t.start()
            return httpd, port
        except Exception:
            continue
    return None, port

if __name__ == "__main__":
    print(f"Starting RetailFlow Portal Server on http://127.0.0.1:8000...")
    httpd, port = start_portal_server(8000)
    if httpd:
        print(f"Portal server is LIVE on http://127.0.0.1:{port}")
        print("Press Ctrl+C to terminate.")
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("Stopping portal server...")
            httpd.shutdown()
    else:
        print(f"Port {port} could not be bound (already in use).")
