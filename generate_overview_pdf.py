import os
import sys
import subprocess

def install_dependencies():
    try:
        import fpdf
    except ImportError:
        print("Installing fpdf2 library...")
        subprocess.run([sys.executable, "-m", "pip", "install", "fpdf2"], check=True)

# Install fpdf2 if not available
install_dependencies()

from fpdf import FPDF

class OverviewPDF(FPDF):
    def header(self):
        # Header text
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(100, 100, 100)
        self.cell(0, 10, "Retail Heatmap Analytics using CCTV -- Project Overview Guide", align="R", new_x="LMARGIN", new_y="NEXT")
        self.set_draw_color(200, 200, 200)
        self.line(20, 18, 190, 18)
        self.ln(5)

    def footer(self):
        # Footer text
        self.set_y(-15)
        self.set_draw_color(220, 220, 220)
        self.line(20, self.get_y() - 2, 190, self.get_y() - 2)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(128, 128, 128)
        self.cell(0, 10, f"Page {self.page_no()} of {{nb}}", align="C")

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    pdf_path = os.path.join(script_dir, "docs", "project_overview.pdf")
    
    html_content = """
    <h1>Retail Heatmap Analytics using CCTV<br><font size="12" color="#64748b">Project Overview & Architecture Guide</font></h1>
    <hr>
    
    <h2>1. Project Goal & Core Concept</h2>
    <p>
        This project is a complete <b>end-to-end Computer Vision and Retail Intelligence Dashboard</b>. It is designed to take video feeds from standard store CCTV cameras, detect shoppers, track their movements, compute how long they browse in specific zones (Aisle A, Aisle B, Electronics, Checkouts), and render a dynamic, color-coded heat map overlays showing high-traffic floor areas. 
    </p>
    
    <h2>2. Key Features</h2>
    <ul>
        <li><b>AI Human Detection:</b> Uses <b>YOLOv8 (You Only Look Once, v8)</b> to locate customers on each frame, filtering out non-human clutter.</li>
        <li><b>Shopper Tracking:</b> Employs <b>ByteTrack</b> algorithms to associate detections across frames, assign persistent unique IDs to each person, and trace visual movement trails.</li>
        <li><b>Dynamic Heatmap Overlay:</b> Generates a 2D density grid of customer positions, applies <b>Gaussian Blur</b> to smooth hot-spots, and overlays them onto the video frame using OpenCV colormaps (Jet, Hot, Rainbow).</li>
        <li><b>Retail Dashboard:</b> Built with a custom, light-themed <b>Streamlit</b> interface featuring live metrics cards (Total Footfall, Active Shoppers, Average Dwell, Busiest Section) and statistical charts (Timeline area-chart, bar graphs, and dwell distribution histograms).</li>
        <li><b>Dual Mode Running:</b> Runs on heavy GPU deep learning (Webcam / Uploaded CCTV video) OR a pre-recorded mock Simulation mode (CPU-only, running at 30 FPS using CSV data).</li>
    </ul>

    <h2>3. Codebase File Structure & Core Classes</h2>
    <ul>
        <li><b>app.py:</b> The user interface portal. Manages the Streamlit execution loop, side-by-side viewports (Live CCTV Feed and Heatmap Visualization), and metric cards rendering.</li>
        <li><b>utils/tracker_wrapper.py (CCTVTracker):</b> The analytical brain. Controls the YOLOv8 model loading, coordinate zone containment checks, trail histories, and customer statistical pandas dataframe compiling.</li>
        <li><b>utils/heatmap_generator.py (HeatmapAccumulator):</b> Manages coordinate density grids. Performs normalization, Gaussian blur kernel applications, colormap coloring, and transparent weighted BGR frame blending.</li>
        <li><b>utils/synthetic_generator.py (generate_video):</b> A mock engine that pre-generates a sample CCTV AVI/MP4 video along with pre-calculated traveler CSV detections. Useful for demonstrating the project instantly without GPU or real surveillance files.</li>
        <li><b>run.py:</b> A unified project launcher that checks and auto-installs system dependencies, checks the synthetic video file, and boots Streamlit locally.</li>
        <li><b>style.css:</b> Global stylesheet setting up clean typography, custom metrics card layouts, and navigation headers.</li>
    </ul>

    <h2>4. Deliverables Package</h2>
    <p>
        The project includes academic documents in the <b>docs/</b> directory to prepare for project presentation:
    </p>
    <ul>
        <li><b>docs/mini_project_report.md:</b> A structured 9-section report (Abstract, Literature Review, System Specifications, Design Flow, Results, and References).</li>
        <li><b>docs/ppt_content.md:</b> A slide-by-slide script for preparing a 15-slide project deck.</li>
        <li><b>docs/viva_questions.md:</b> A database of 25+ Viva prep questions with answers covering algorithms, calculations, and performance metrics.</li>
        <li><b>docs/system_architecture.mermaid:</b> The flowchart mapping components and data streams.</li>
    </ul>
    """
    
    # Wrap in standard HTML structure
    full_html = f"""
    <html>
    <head>
    <style>
        body {{
            font-family: Helvetica, Arial, sans-serif;
            font-size: 10pt;
            line-height: 1.5;
            color: #111111;
        }}
        h1 {{
            font-size: 18pt;
            font-weight: bold;
            color: #2563eb;
            text-align: center;
            margin-bottom: 15px;
        }}
        h2 {{
            font-size: 12pt;
            font-weight: bold;
            color: #0f172a;
            margin-top: 15px;
            margin-bottom: 6px;
            border-bottom: 1px solid #e2e8f0;
            padding-bottom: 3px;
        }}
        p {{
            margin-bottom: 10px;
            text-align: justify;
        }}
        ul {{
            margin-bottom: 10px;
            margin-left: 20px;
        }}
        li {{
            margin-bottom: 6px;
        }}
        hr {{
            border: 0;
            border-top: 1px solid #cbd5e1;
            margin: 10px 0;
        }}
    </style>
    </head>
    <body>
    {html_content}
    </body>
    </html>
    """
    
    print("Generating Project Overview PDF...")
    pdf = OverviewPDF()
    pdf.set_margins(20, 20, 20)
    pdf.add_page()
    pdf.set_font("Helvetica", size=10)
    
    try:
        pdf.write_html(full_html)
    except Exception as e:
        print(f"Warning during HTML parsing: {e}")
        pdf.add_page()
        pdf.set_font("Courier", size=9)
        pdf.write(5, html_content)
        
    print(f"Saving PDF to {pdf_path}...")
    pdf.output(pdf_path)
    print("Overview PDF generated successfully!")

if __name__ == "__main__":
    main()
