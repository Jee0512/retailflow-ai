import os
import sys
import subprocess

def check_and_install_dependencies():
    """Checks if required packages are installed, and installs them if missing."""
    required_packages = {
        "ultralytics": "ultralytics",
        "streamlit": "streamlit",
        "cv2": "opencv-python",
        "numpy": "numpy",
        "pandas": "pandas",
        "matplotlib": "matplotlib",
        "seaborn": "seaborn",
        "fpdf": "fpdf2",
        "plotly": "plotly"
    }
    
    missing_packages = []
    for module_name, pip_name in required_packages.items():
        try:
            __import__(module_name)
        except ImportError:
            missing_packages.append(pip_name)
            
    if missing_packages:
        print("=" * 60)
        print(f"Missing dependencies found: {', '.join(missing_packages)}")
        print("Installing dependencies via pip. Please wait...")
        print("=" * 60)
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", *missing_packages])
            print("Dependencies installed successfully!\n")
        except subprocess.CalledProcessError as e:
            print(f"Error installing dependencies: {e}")
            print("Please run: pip install -r requirements.txt manually.")
            sys.exit(1)
    else:
        print("All dependencies are already satisfied.")

def generate_sample_video_if_needed():
    """Generates the sample CCTV simulation video if it does not exist."""
    video_path = "sample_cctv.mp4"
    if not os.path.exists(video_path):
        print("=" * 60)
        print("Generating synthetic retail CCTV footage for initial demo...")
        print("=" * 60)
        
        # Ensure utils folder exists and has __init__.py
        os.makedirs("utils", exist_ok=True)
        if not os.path.exists("utils/__init__.py"):
            with open("utils/__init__.py", "w") as f:
                pass
                
        # Run the synthetic generator script
        try:
            from utils.synthetic_generator import generate_video
            generate_video(video_path, num_frames=900, width=1280, height=720)
            print(f"Successfully generated sample CCTV video: {video_path}\n")
        except Exception as e:
            print(f"Failed to generate synthetic video: {e}")
            print("Ensure utils/synthetic_generator.py is implemented and functional.")
            sys.exit(1)
    else:
        print(f"Sample video '{video_path}' already exists. Skipping generation.")

def main():
    print("=" * 60)
    print("    Retail Heatmap Analytics using CCTV - Project Launcher    ")
    print("=" * 60)
    
    # 1. Dependency check
    check_and_install_dependencies()
    
    # 2. Sample video check/generation
    generate_sample_video_if_needed()
    
    # 3. Initialize SQLite auth database
    try:
        import auth_db
        auth_db.init_db()
        print("SQLite Authentication Database (users.db) is initialized and active.")
    except Exception as e:
        print(f"Auth DB notice: {e}")

    # 3b. Start background HTTP Portal Server (Home Landing & Admin Console)
    try:
        from portal_server import start_portal_server
        httpd, portal_port = start_portal_server(8000)
        print(f"HTTP Portal Server running on http://127.0.0.1:{portal_port}")
    except Exception as e:
        print(f"Portal server notice: {e}")

    # 4. Launch Streamlit and open Home Portal in Browser
    import webbrowser
    import threading
    import time

    def open_portal_in_browser():
        time.sleep(1.5)
        try:
            webbrowser.open("http://127.0.0.1:8000")
        except Exception:
            pass

    threading.Thread(target=open_portal_in_browser, daemon=True).start()

    print("=" * 60)
    print("Launching RetailFlow.AI System...")
    print("🏠 Home Portal:        http://127.0.0.1:8000")
    print("🛡️ Admin Console:      http://127.0.0.1:8000/admin.html")
    print("📊 Streamlit Dashboard: http://127.0.0.1:8501")
    print("Opening Home Portal in default browser...")
    print("Press Ctrl+C to terminate.")
    print("=" * 60)
    
    try:
        # Launch Streamlit reliably using sys.executable -m streamlit
        subprocess.run([sys.executable, "-m", "streamlit", "run", "app.py"], check=True)
    except KeyboardInterrupt:
        print("\nProject launcher stopped by user.")
    except Exception as e:
        print(f"\nError launching dashboard: {e}")
        print("Make sure you can run: streamlit run app.py")

if __name__ == "__main__":
    main()
