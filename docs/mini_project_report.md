# Academic Mini Project Report

**Project Title:** Retail Heatmap Analytics using CCTV Smart Surveillance  
**Academic Year:** 2026  
**Course:** Bachelor of Technology / Bachelor of Engineering (Computer Science & Engineering)  
**Project Category:** Computer Vision, Deep Learning, Retail Business Intelligence  

---

## 1. Abstract
In modern retail, understanding customer behavior, navigation patterns, and shopping shelf interaction is vital for space optimization, marketing strategy, and layout designing. While traditional methods rely on manual counting or RFID trackers, they are expensive and fail to capture full customer trajectories. This project presents a **Retail Heatmap Analytics System** that uses computer vision and deep learning on standard store CCTV feeds. The system detects shoppers using **YOLOv8** (You Only Look Once, Version 8) and tracks their paths using a robust object tracker. It dynamically maps customer paths into a coordinate accumulator to generate a real-time visual density heatmap using Gaussian smoothing and OpenCV blending techniques. Dwell times are computed for distinct store zones (Aisle A, Aisle B, Checkouts) and visual reports are served via an interactive, dark-themed **Streamlit** dashboard. To support low-latency demonstration on standard laptops, the system includes a synthetic shopper simulation engine.

---

## 2. Introduction
Surveillance cameras are ubiquitously deployed in retail spaces, but their use is largely restricted to security auditing. Using this pre-existing infrastructure for commercial analytics creates a massive opportunity for brick-and-mortar retail stores to gain digital analytics comparable to e-commerce tracking (like Google Analytics).

### 2.1 Objectives
- **Automate Footfall Counting**: Replace manual head-counting with automated human detection.
- **Understand Navigation Patterns**: Draw customer trails to identify common walking routes.
- **Generate Density Heatmaps**: Highlight "hot spots" (high traffic shelves) and "dead zones" (unvisited areas) in the store.
- **Measure Shelf Attention (Dwell Time)**: Calculate exactly how long customers browse specific product categories.
- **Build Retailer HUD**: Create an intuitive visual interface with charts showing visitor flows.

---

## 3. Literature Review
Earlier surveillance analytics systems relied heavily on basic **Background Subtraction** algorithms (e.g., Gaussian Mixture Models) or **Haar Cascades** for person detection. These approaches suffered from extreme sensitivity to illumination changes, shadows, and occlusion. 
With the advent of deep Convolutional Neural Networks, real-time detectors like **YOLO** achieved state-of-the-art accuracy and frame rates. Standard tracking frameworks like **DeepSORT** (Simple Online and Realtime Tracking with a Deep Association Metric) integrated Kalman Filters and deep appearance descriptors to associate people across frames. Our project implements YOLOv8 and tracking heuristics to solve the tracking challenge without high computational overhead.

---

## 4. System Requirements

### 4.1 Hardware Requirements
- **Processor**: Intel Core i5 / AMD Ryzen 5 or higher (Quad-Core minimum)
- **Memory**: 8 GB RAM (16 GB recommended for smooth YOLO execution)
- **Graphics Card**: Dedicated NVIDIA GPU with CUDA support (Optional, but recommended for real-time camera feeds at full FPS)
- **Camera Input**: Standard 720p/1080p USB webcam or RTSP IP Camera stream

### 4.2 Software Requirements
- **Operating System**: Windows 10/11, Linux, or macOS
- **Language**: Python 3.9+
- **Primary Libraries**:
  - `opencv-python`: Image manipulation, colormaps, image drawing, and alpha blending.
  - `ultralytics`: YOLOv8 object detection and ByteTrack wrapper.
  - `streamlit`: Analytical dashboard layout and web interface.
  - `pandas` & `numpy`: Multi-dimensional array handling and statistical calculations.
  - `matplotlib` & `seaborn`: Analytical graph plotting.

---

## 5. Methodology & System Architecture

The project consists of four main pipelines as outlined below:

```
[ CCTV Video Input ] 
       │
       ▼
[ Preprocessing (Resize/Frame Read) ] 
       │
       ▼
[ YOLOv8 Person Detector ] ---> (Filters other classes, retains 'Person')
       │
       ▼
[ Tracking Module (ByteTrack) ] ---> (Assigns unique IDs & maintains trails)
       │
 ┌─────┴────────────────────────────────────────┐
 │                                              │
 ▼                                              ▼
[ Heatmap Generation Pipeline ]       [ Analytics Compilation Pipeline ]
 - Accumulate (x,y) centers            - Map centers to store zone bounds
 - Apply Gaussian Blur filter          - Sum frame durations in each zone
 - Colormap mapping (Jet/Hot)          - Calculate dwell times (Frames/FPS)
 - Alpha blend onto background         - Store logs in Pandas DataFrame
 │                                              │
 └─────┬────────────────────────────────────────┘
       │
       ▼
[ Streamlit UI Dashboard ]
 - Live CCTV bounding box display
 - Live Heatmap overlay display
 - Real-time metrics (Footfall, Dwell, Busy Zone)
 - Visual analytics charts (footfall flow, bar graphs)
 - CSV downloader for reports
```

### 5.1 Person Detection (YOLOv8)
The frames are fed into YOLOv8. The network parses the frame features and yields bounding box coordinates `[x_min, y_min, x_max, y_max]`, confidence scores, and class labels. The system discards all class labels except `Person (Class ID 0)`.

### 5.2 Customer Tracking & Trails
To prevent duplicate counting when people move or temporarily exit the view, a tracking algorithm assigns unique numbers to each detected box based on overlap intersection-over-union (IoU) and motion predictions (Kalman Filter). The centroids `(cx, cy)` of these boxes are calculated and stored in a sliding-history queue to draw trailing paths.

### 5.3 Staff Detection & Analytics Exclusion
In real-world retail store analytics, store associates walking aisles or stationed at registers can severely skew customer dwell and traffic data. To resolve this, our system introduces a Computer Vision uniform classification pipeline:
1. **Torso Extraction**: Bounding boxes are cropped to the upper torso $[y_1 + 0.15h : y_1 + 0.60h, x_1 + 0.20w : x_2 - 0.20w]$.
2. **HSV Uniform Masking**: The torso is evaluated in HSV color space against configured uniform color boundaries (e.g., Red, Blue, Green/Neon, Black, Orange, or Custom Hex).
3. **Temporal Voting**: A rolling window of classification votes ensures identity stability despite variations in store lighting.
4. **Data Isolation**: Detected staff are tagged with an amber badge `[STAFF #ID (Filtered)]`. Their coordinates are excluded from the heatmap accumulator and omitted from customer footfall and zone dwell computations.

### 5.4 Heatmap Overlay
The coordinate centers of tracked customers are mapped to a 2D float matrix of the same resolution as the video. Points are accumulated. The matrix is smoothed using a Gaussian kernel:
$$G(x, y) = \frac{1}{2\pi\sigma^2} e^{-\frac{x^2 + y^2}{2\sigma^2}}$$
A color map is applied to map low-value regions to cool colors (blue) and high-value regions to warm colors (red). The heatmap is overlaid onto the original frame using BGR blending:
$$\text{Output} = (1 - \alpha) \cdot \text{BaseFrame} + \alpha \cdot \text{ColormappedHeatmap}$$

### 5.5 Zone-Wise Analytics
A coordinate containment check verifies if a shopper's center `(cx, cy)` falls inside predetermined rectangles:
$$x_1 \le cx \le x_2 \quad \text{and} \quad y_1 \le cy \le y_2$$
Dwell time in seconds is calculated as:
$$\text{Dwell Time} = \frac{\text{Frames occupied in zone}}{\text{Video Frame Rate (FPS)}}$$

---

## 6. Implementation Details

The codebase is split into modular components:
1. `app.py`: Streamlit main dashboard. Manages UI elements, sidebars, charts, layout, and processing loops.
2. `utils/tracker_wrapper.py`: Manages model loading, object detection, ID tracking, zone checks, and customer metrics dataframe compiling.
3. `utils/heatmap_generator.py`: Manages coordinate density tracking, Gaussian blur, color mapping, and blending overlays.
4. `utils/synthetic_generator.py`: Generates the mock simulation video with simulated shopper agents following retail navigation path rules. This makes the project functional instantly without requiring external database setups.

---

## 7. Results & Analysis
- **Demo Mode**: Yields a clean, smooth 30 FPS playback simulating typical customer flow. Bounding boxes are accurately updated and trails are drawn. The heat accumulation highlights the Produce and Electronics Aisles and Checkout queues.
- **YOLOv8 Mode**: Successfully detects humans from uploaded surveillance files. The ByteTrack algorithm maintains ID consistency during partial occlusions.
- **Analytics Outputs**: Charts show peak traffic flow time and bar charts highlight the division of shopper attention across zones. The CSV report matches the visual output exactly.

---

## 8. Conclusion & Future Scope
This project successfully demonstrates an automated, cost-effective retail analytics platform using CCTV surveillance footage. By using deep learning and mathematical image blending, the system tracks footfall, plots visitor maps, and logs shopper attention time.

### 8.1 Future Scope
1. **Multi-Camera Association**: Stitching tracks across multiple store cameras so a customer's ID is maintained as they walk between different rooms.
2. **Staff/Customer Differentiation**: Recognizing employees by uniform color histograms or facial templates and removing them from marketing analytics.
3. **Queue Prediction**: Triggering alert notifications for store managers when checkout queue length exceeds 5 shoppers.
4. **Behavior Analysis**: Integrating pose estimation to detect shelf-grabbing motions or suspicious retail-theft behaviors.

---

## 9. References
1. Redmon, J., et al. "You Only Look Once: Unified, Real-Time Object Detection." *CVPR 2016*.
2. Bewley, A., et al. "Simple Online and Realtime Tracking." *ICIP 2016*.
3. Ultralytics YOLOv8 Documentation (https://docs.ultralytics.com).
4. Streamlit Dashboard API Reference (https://docs.streamlit.io).
