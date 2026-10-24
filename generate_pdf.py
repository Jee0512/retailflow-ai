import os
import sys
import re
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

def sanitize_for_latin1(text):
    # Map characters that are outside latin-1 to clean ascii equivalents
    replacements = {
        "\u2014": "--",
        "\u2013": "-",
        "\u201c": '"',
        "\u201d": '"',
        "\u2018": "'",
        "\u2019": "'",
        "\u2022": "*",
        "\u2192": "-->",
        "\u2500": "-",
        "\u2502": "|",
        "\u250c": "+",
        "\u2510": "+",
        "\u2514": "+",
        "\u2518": "+",
        "\u251c": "+",
        "\u2524": "+",
        "\u252c": "+",
        "\u2534": "+",
        "\u253c": "+",
        "\u25bc": "v",
        "\u2190": "<--",
        "\u2191": "^",
        "\u2193": "v",
    }
    for char, replacement in replacements.items():
        text = text.replace(char, replacement)
    
    # Encode/decode check to strip out any un-representable characters
    safe_chars = []
    for char in text:
        try:
            char.encode("latin-1")
            safe_chars.append(char)
        except UnicodeEncodeError:
            safe_chars.append("?")
    return "".join(safe_chars)

class AcademicReportPDF(FPDF):
    def header(self):
        # Academic Header
        if self.page_no() > 1:
            self.set_font("Helvetica", "I", 8)
            self.set_text_color(100, 100, 100)
            header_text = sanitize_for_latin1("Retail Heatmap Analytics using CCTV Smart Surveillance -- Academic Report")
            self.cell(0, 10, header_text, align="R", new_x="LMARGIN", new_y="NEXT")
            self.set_draw_color(200, 200, 200)
            self.line(20, 18, 190, 18) # draw thin divider line below header
            self.ln(5)

    def footer(self):
        # Academic Footer
        self.set_y(-15)
        self.set_draw_color(220, 220, 220)
        self.line(20, self.get_y() - 2, 190, self.get_y() - 2)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(128, 128, 128)
        self.cell(0, 10, f"Page {self.page_no()} of {{nb}}", align="C")

def clean_latex(latex_str):
    # Replace LaTeX operators with clean Unicode strings for plaintext rendering
    replacements = [
        (r"\frac{1}{2\pi\sigma^2}", "1 / (2 * pi * sigma^2)"),
        (r"e^{-\frac{x^2 + y^2}{2\sigma^2}}", "e^(-(x^2 + y^2) / (2 * sigma^2))"),
        (r"G(x, y) = ", "G(x, y) = "),
        (r"\text{Output} = ", "Output = "),
        (r"\text{BaseFrame}", "BaseFrame"),
        (r"\text{ColormappedHeatmap}", "ColormappedHeatmap"),
        (r"\text{Dwell Time} = ", "Dwell Time = "),
        (r"\text{Frames occupied in zone}", "Frames occupied in zone"),
        (r"\text{Video Frame Rate (FPS)}", "Video Frame Rate (FPS)"),
        (r"\cdot", " * "),
        (r"\alpha", "alpha"),
        (r"\sigma", "sigma"),
        (r"\pi", "pi"),
        (r"\le", " <= "),
        (r"\quad", " "),
        (r"\text{and}", "and"),
    ]
    result = latex_str
    for pat, rep in replacements:
        result = result.replace(pat, rep)
    return result.strip()

def markdown_to_html(md_text):
    lines = md_text.split("\n")
    html_out = []
    in_code_block = False
    code_lines = []
    
    in_list = False
    list_type = None # 'ul' or 'ol'
    
    for line in lines:
        stripped = line.strip()
        
        # 1. Code blocks (Preformatted text)
        if stripped.startswith("```"):
            if in_code_block:
                in_code_block = False
                # Close code block
                code_content = "\n".join(code_lines)
                # Sanitize block for latin-1
                code_content = sanitize_for_latin1(code_content)
                code_content = code_content.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
                html_out.append(f"<pre style='font-family: Courier; font-size: 8pt;'>{code_content}</pre>")
                code_lines = []
            else:
                in_code_block = True
            continue
            
        if in_code_block:
            code_lines.append(line)
            continue
            
        # 2. Horizontal Rules
        if stripped == "---":
            if in_list:
                html_out.append(f"</{list_type}>")
                in_list = False
            html_out.append("<hr>")
            continue
            
        # 3. Headings
        heading_match = re.match(r"^(#{1,6})\s+(.*)$", line)
        if heading_match:
            if in_list:
                html_out.append(f"</{list_type}>")
                in_list = False
            level = len(heading_match.group(1))
            title = heading_match.group(2)
            title = process_inline_styles(title)
            html_out.append(f"<h{level}>{title}</h{level}>")
            continue
            
        # 4. Bullet lists
        bullet_match = re.match(r"^-\s+(.*)$", line)
        if bullet_match:
            if not in_list or list_type != "ul":
                if in_list:
                    html_out.append(f"</{list_type}>")
                html_out.append("<ul>")
                in_list = True
                list_type = "ul"
            content = process_inline_styles(bullet_match.group(1))
            html_out.append(f"<li>{content}</li>")
            continue
            
        # 5. Numbered lists
        num_match = re.match(r"^(\d+)\.\s+(.*)$", line)
        if num_match:
            if not in_list or list_type != "ol":
                if in_list:
                    html_out.append(f"</{list_type}>")
                html_out.append("<ol>")
                in_list = True
                list_type = "ol"
            content = process_inline_styles(num_match.group(2))
            html_out.append(f"<li>{content}</li>")
            continue
            
        # If line is empty
        if not stripped:
            if in_list:
                html_out.append(f"</{list_type}>")
                in_list = False
            continue
            
        # 6. Regular paragraph line
        if in_list:
            html_out.append(f"</{list_type}>")
            in_list = False
            
        processed_line = process_inline_styles(line)
        # Check if line ends with two spaces (markdown line break)
        if line.endswith("  "):
            html_out.append(f"{processed_line}<br>")
        else:
            html_out.append(processed_line)
            
    if in_list:
        html_out.append(f"</{list_type}>")
        
    # Now group paragraph text lines
    final_html = []
    current_p = []
    
    for item in html_out:
        is_html_tag = (
            item.startswith("<h") or 
            item.startswith("</h") or 
            item.startswith("<ul") or 
            item.startswith("</ul") or 
            item.startswith("<ol") or 
            item.startswith("</ol") or 
            item.startswith("<li") or 
            item.startswith("</li") or 
            item.startswith("<pre") or 
            item.startswith("<hr") or 
            item == "<br>"
        )
        
        if is_html_tag:
            if current_p:
                final_html.append("<p>" + " ".join(current_p) + "</p>")
                current_p = []
            final_html.append(item)
        else:
            current_p.append(item)
            
    if current_p:
        final_html.append("<p>" + " ".join(current_p) + "</p>")
        
    return "\n".join(final_html)

def process_inline_styles(text):
    text = sanitize_for_latin1(text)
    
    # Escape math formulas to make them look readable in text
    # Block math
    text = re.sub(r"\$\$(.*?)\$\$", lambda m: f"<br> &nbsp;&nbsp;&nbsp;&nbsp;<b>{clean_latex(m.group(1))}</b><br>", text)
    # Inline math
    text = re.sub(r"\$(.*?)\$", lambda m: f"<b>{clean_latex(m.group(1))}</b>", text)
    
    # Bold text
    text = re.sub(r"\*\*(.*?)\*\*", r"<b>\1</b>", text)
    
    # Italic text
    text = re.sub(r"\*(.*?)\*", r"<i>\1</i>", text)
    
    # Links
    text = re.sub(r"\[(.*?)\]\((.*?)\)", r'<a href="\2">\1</a>', text)
    
    return text

def generate_academic_report(script_dir):
    md_path = os.path.join(script_dir, "docs", "mini_project_report.md")
    pdf_path = os.path.join(script_dir, "docs", "mini_project_report.pdf")
    
    if not os.path.exists(md_path):
        print(f"Error: Markdown file not found at {md_path}")
        return None
        
    print(f"[1/3] Compiling Academic Project Report from {md_path}...")
    with open(md_path, "r", encoding="utf-8") as f:
        md_content = f.read()
        
    html_content = markdown_to_html(md_content)
    
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
            color: #1a0dab;
            text-align: center;
            margin-bottom: 20px;
        }}
        h2 {{
            font-size: 13pt;
            font-weight: bold;
            color: #0f172a;
            margin-top: 15px;
            margin-bottom: 8px;
            border-bottom: 1px solid #e2e8f0;
            padding-bottom: 4px;
        }}
        h3 {{
            font-size: 11pt;
            font-weight: bold;
            color: #334155;
            margin-top: 10px;
            margin-bottom: 6px;
        }}
        p {{
            margin-bottom: 10px;
            text-align: justify;
        }}
        ul, ol {{
            margin-bottom: 10px;
            margin-left: 20px;
        }}
        li {{
            margin-bottom: 4px;
        }}
        pre {{
            font-family: Courier, monospace;
            background-color: #f8fafc;
            border: 1px solid #e2e8f0;
            padding: 8px;
            margin-bottom: 10px;
        }}
        hr {{
            border: 0;
            border-top: 1px solid #cbd5e1;
            margin: 15px 0;
        }}
    </style>
    </head>
    <body>
    {html_content}
    </body>
    </html>
    """
    
    pdf = AcademicReportPDF()
    pdf.set_margins(20, 20, 20)
    pdf.add_page()
    pdf.set_font("Helvetica", size=10)
    
    try:
        pdf.write_html(full_html)
    except Exception as e:
        print(f"Warning during HTML parsing: {e}")
        pdf.add_page()
        pdf.set_font("Courier", size=8)
        clean_md = sanitize_for_latin1(md_content)
        pdf.write(5, clean_md)
        
    pdf.output(pdf_path)
    print(f"      [OK] Academic Report saved to: {pdf_path}")
    return pdf_path


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    print("=" * 65)
    print("  RetailFlow.AI - Unified PDF Document Generation Suite")
    print("=" * 65)

    # 1. Store Executive Performance Report (Root PDF)
    print("[1/3] Generating Executive Store Performance Report (Retail_Store_Performance_Report.pdf)...")
    try:
        from report_generator import generate_sample_report
        store_report_path, data = generate_sample_report()
        print(f"      [OK] Store Performance Report: {store_report_path} ({len(data):,} bytes)")
    except Exception as e:
        print(f"      [ERROR] Failed to build store performance report: {e}")

    # 2. Academic Mini Project Report
    try:
        generate_academic_report(script_dir)
    except Exception as e:
        print(f"      [ERROR] Failed to build academic report: {e}")

    # 3. Project Overview Architecture Guide
    print("[3/3] Generating Project Overview Guide (docs/project_overview.pdf)...")
    try:
        from generate_overview_pdf import main as gen_overview
        gen_overview()
        print(f"      [OK] Project Overview Guide saved in docs/")
    except Exception as e:
        print(f"      [ERROR] Failed to build project overview: {e}")

    print("=" * 65)
    print("  All PDF documents generated successfully!")
    print("=" * 65)


if __name__ == "__main__":
    main()

