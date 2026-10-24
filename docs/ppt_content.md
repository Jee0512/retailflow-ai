# PowerPoint Presentation Content (PPT Slides Outline)

This document provides a slide-by-slide layout for a final-year project presentation. Use this content to populate your PowerPoint or Google Slides deck.

---

## Slide 1: Title Slide
* **Slide Title:** Retail Heatmap Analytics using CCTV Smart Surveillance
* **Subtitle:** Deep Learning & Computer Vision for Retail Business Intelligence
* **Content:**
  * **Presented By:** [Student Name 1], [Student Name 2], [Student Name 3]
  * **Under the Guidance of:** [Advisor/Professor Name]
  * **Institution:** [College/University Name]
  * **Department:** Computer Science & Engineering / Information Technology

---

## Slide 2: Motivation & Background
* **Slide Title:** Project Motivation
* **Content:**
  * **The Problem:** Brick-and-mortar retail stores lack the detailed customer analytics (paths, dwell times, and attention mapping) that e-commerce stores easily track.
  * **The Opportunity:** Modern retail outlets already have CCTV security cameras installed.
  * **The Solution:** Leverage computer vision to convert standard security footage into a marketing intelligence tool without purchasing new hardware.
  * **Key Business Value:** Optimize shelf layouts, calculate ROI on displays, and improve checkout staffing.

---

## Slide 3: Problem Statement
* **Slide Title:** Problem Statement
* **Content:**
  * To design and implement a real-time computer vision system that:
    1. Detects and tracks customers moving through a retail space under variable lighting.
    2. Identifies high-traffic areas ("hot spots") and low-traffic areas ("dead zones").
    3. Measures customer interest in specific shelves by tracking dwell times.
    4. Displays visual analytics, historical trends, and exports data via an intuitive user interface.

---

## Slide 4: System Architecture
* **Slide Title:** System Architecture & Workflow
* **Content:**
  * **Input Layer:** CCTV RTSP/Video Stream -> OpenCV Frame Reader.
  * **Detection & Tracking:** YOLOv8 detects humans -> ByteTrack associates bounding boxes to unique IDs -> Coordinate extraction.
  * **Heatmap Generator:** Coordinates mapped to 2D accumulator grid -> Gaussian Blur smoothing -> Overlay blended on frame.
  * **Analytics Engine:** Zone containment calculation -> Frame counter divided by FPS = Dwell Time.
  * **Presentation Dashboard:** Streamlit UI displays CCTV frame, heatmap, metric cards, charts, and report exports.

---

## Slide 5: Human Detection (YOLOv8)
* **Slide Title:** Object Detection using YOLOv8
* **Content:**
  * **Why YOLOv8?** State-of-the-art accuracy, optimized neural architecture, and fast real-time inference on normal CPU-based laptops.
  * **Filtering Logic:** Bounding boxes are filtered by confidence (e.g. > 35%) and Class ID.
  * **Class Filtering:** Only "Person" (Class ID 0) is sent to the tracking module; shopping carts, products, and bags are ignored.
  * **Output:** Yields normalized coordinates `[x1, y1, x2, y2]` for each shopper.

---

## Slide 6: Customer Tracking & Trails
* **Slide Title:** Multi-Object Tracking & Motion Trails
* **Content:**
  * **Tracking Algorithm:** Integrated ByteTrack/BoT-SORT tracker.
  * **Key Logic:**
    * Assigns stable, unique IDs to shoppers.
    * Associates bounding boxes across frames using Kalman Filters and IoU matching.
    * Handles brief occlusions (e.g. customer walking behind a pillar or shelf).
  * **Motion Trails:**
    * Centroid coordinates are logged in a queue.
    * Line vectors connect historical centroids to visualize paths.
    * Fading effect prevents visual clutter in high-footfall cameras.

---

## Slide 7: Heatmap Generation Module
* **Slide Title:** Dynamic Traffic Heatmaps
* **Content:**
  * **Accumulator Array:** A float matrix initialized to match the video resolution.
  * **Smoothing Filter:**
    * Coordinates increment the matrix values.
    * Gaussian Blur is applied dynamically to smooth out sparse footprints.
  * **Visual Blending:**
    * Convert density value to color using BGR colormaps (e.g., Jet/Hot colormaps).
    * BGR blending mixes the heatmap with the original frame:  
      `Output = (1 - alpha) * BaseFrame + alpha * HeatmapColor`
    * Masking ensures areas with zero footfall remain uncolored.

---

## Slide 8: Zone-Based Dwell Time Analytics
* **Slide Title:** Section Popularity & Attention Tracking
* **Content:**
  * **Aisle Boundaries:** Coordinate bounding boxes define custom zones (Produce, Apparel, Electronics, Checkouts).
  * **Containment Logic:** Check if shopper coordinate `(cx, cy)` lies within zone boundaries.
  * **Dwell Time Accumulator:**
    * Count frames spent by an ID in a zone.
    * `Dwell Time = Frames / Video FPS`.
  * **Retail Value:** Provides concrete numbers on which products attract the most attention.

---

## Slide 9: User Interface Dashboard
* **Slide Title:** Streamlit Web Application
* **Content:**
  * **Design Philosophy:** Dark theme, clean layouts, and glowing indicators.
  * **Live Stream Controls:** Start/Stop/Pause CCTV video streams.
  * **Dynamic Metric Cards:** Show Live Footfall, Avg Dwell Time, Active Shoppers, Busiest Store Zone.
  * **Dual Monitor Views:** Side-by-side display of live bounding-box tracking and the density heatmap.
  * **Export Option:** One-click download of shopper analytics reports as a CSV file.

---

## Slide 10: Technical Stack
* **Slide Title:** System Technology Stack
* **Content:**
  * **Language:** Python
  * **Deep Learning Framework:** Ultralytics YOLOv8 (PyTorch back-end)
  * **Computer Vision Library:** OpenCV (Open Source Computer Vision)
  * **Data Wrangling:** Pandas & NumPy
  * **Plotting & Graphs:** Matplotlib & Seaborn
  * **Front-end Web Framework:** Streamlit (Python-native dashboards)

---

## Slide 11: Project Timeline & Task Division
* **Slide Title:** Project Management & Team Work Division
* **Content:**
  * **Member 1 (AI & CV Engineer):** YOLOv8 setup, tracking algorithm integration, and path-trail plotting.
  * **Member 2 (Image Processing Engineer):** Heatmap coordinate accumulator, Gaussian blur logic, colormap blending, and synthetic video generator.
  * **Member 3 (Dashboard & Analytics Developer):** Streamlit layout, Matplotlib/Seaborn integration, metrics calculation, database logger, and report exporter.
  * **Completion Timeline:** Total duration: 8 Weeks (from requirements analysis to final evaluation).

---

## Slide 12: Future Scope & Enhancements
* **Slide Title:** Future Directions
* **Content:**
  * **Multi-Camera Re-Identification:** Track customers moving across different camera angles throughout the entire store.
  * **Demographic Analytics:** Integrate light gender and age classification models.
  * **Manager Notifications:** Automatically alert staff on mobile/slack when checkout queue lengths cross a threshold.
  * **Behavioral Analysis:** Detect shoplifting patterns (suspicious movement, hiding items) using pose estimation.

---

## Slide 13: Conclusion
* **Slide Title:** Conclusion
* **Content:**
  * Successfully developed an end-to-end Retail CCTV analytics system.
  * System executes on consumer-grade hardware/laptops.
  * Replaces expensive RFID or BLE beacon tracking with existing surveillance hardware.
  * Delivers highly actionable insights to retail store owners.

---

## Slide 14: Q&A / Questions?
* **Slide Title:** Questions & Answers
* **Content:**
  * Thank you for your time.
  * Open for questions and feedback.
