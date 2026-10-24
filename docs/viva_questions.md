# Viva Voce Questions & Answers

This guide contains 25+ high-quality viva questions and answers tailored for this project to prepare team members for project exhibitions and examinations.

---

### Part 1: Deep Learning & YOLOv8 Questions

#### Q1: What is YOLO? How is it different from traditional object detectors like R-CNN?
**Answer:** YOLO stands for "You Only Look Once". It is a **single-stage detector**, meaning it frames object detection as a single regression problem, predicting bounding boxes and class probabilities directly from full images in one evaluation. Traditional models like Faster R-CNN are **two-stage detectors**: they first generate regional proposals (candidate boxes) and then classify those regions. YOLO is much faster, making it suitable for real-time applications like CCTV analysis.

#### Q2: Which version of YOLO is used in this project, and why?
**Answer:** We use **YOLOv8** (released by Ultralytics). We selected it because:
1. It is anchor-free (predicts coordinates directly rather than offset from anchor boxes), improving performance.
2. It uses a new backbone network and decoupled head for higher accuracy and faster inference.
3. It has built-in tracking support (ByteTrack/BoT-SORT) accessible with a clean Python API.

#### Q3: What is "Anchor-Free" detection in YOLOv8?
**Answer:** Older YOLO models (like YOLOv3/v4/v5) used predefined anchor boxes of different aspect ratios. The model predicted offset coordinates relative to these anchors. YOLOv8 is anchor-free; it predicts the distance from the center of an object directly to its bounding box boundaries, reducing hyperparameters and simplifying the loss function.

#### Q4: What does the confidence threshold parameter do?
**Answer:** The confidence threshold represents the minimum probability score a detected object must have to be considered valid by the detector. If the threshold is 0.35, any bounding box with a probability of being a person below 35% is ignored. Raising it reduces false positives (noise), while lowering it catches more people but may introduce false detections (like coat hangers or mannequins).

#### Q5: How do you filter out non-human objects from YOLOv8 detections?
**Answer:** In the COCO dataset (which YOLOv8 is pre-trained on), each class has an index. The `person` class has index `0`. In our code, we tell YOLO's tracking API to filter detections by setting `classes=[0]`. Any detected bounding box that matches other labels (like bags, chairs, or shopping carts) is automatically discarded.

---

### Part 2: Object Tracking & Trails Questions

#### Q6: Why do we need a Tracking algorithm if YOLO already detects people?
**Answer:** YOLO is a **detector**, not a tracker. It processes frames independently. If frame 1 detects a person, and frame 2 detects the same person, YOLO has no concept that they are the same individual; it simply outputs two separate boxes. A **tracker** assigns a persistent ID (e.g., Shopper #4) to the bounding box and maintains that ID across all frames. This is essential to calculate dwell times and avoid duplicate counts.

#### Q7: How does ByteTrack associate detections across frames?
**Answer:** ByteTrack works by associating almost every detection box instead of just high-score boxes. It splits detections into high-confidence and low-confidence boxes.
1. It first matches high-confidence boxes to existing tracks using Intersection over Union (IoU) overlap and motion prediction (Kalman Filters).
2. It then attempts to match the remaining unmatched tracks with low-confidence detections to recover occluded objects (e.g., when a shopper is partially blocked by a pillar).

#### Q8: What is a Kalman Filter?
**Answer:** A Kalman Filter is a mathematical algorithm used to estimate and predict the state of a moving object (position and velocity) over time, even in the presence of noise. In tracking, it predicts where a person's bounding box is expected to be in the next frame based on their past speed and direction.

#### Q9: How are motion trails drawn behind shoppers?
**Answer:** For every unique track ID, we maintain a history list of their bounding box centroids `(cx, cy)`. In each frame, we retrieve the last 25 coordinates of this list and draw connecting lines between them using OpenCV's `cv2.line` function, creating a visual path trail.

#### Q10: How does the system handle ID switching or duplicate counting?
**Answer:** Track association using ByteTrack minimizes ID switching. However, if a shopper exits the camera field of view completely and re-enters, they will be assigned a new ID. This is a standard limitation of single-camera trackers. It can be mitigated by keeping inactive tracks in memory for a short grace period (e.g., 30 frames) before deleting them.

---

### Part 3: Image Processing & Heatmap Questions

#### Q11: What is a Heatmap? How is it represented mathematically in code?
**Answer:** A heatmap represents spatial density. In code, it is represented as a 2D numpy array `accum_map` of floats, initialized to zeros, with the exact width and height of the video frame. Whenever a customer's center point `(cx, cy)` is detected, we increment the pixel values in a small circle around `(cx, cy)` on this array.

#### Q12: Why do we apply Gaussian Blur to the heatmap?
**Answer:** Without Gaussian blur, the accumulated coordinates would just show sharp, jagged circles or isolated dots. Applying a Gaussian Blur smooths the discrete coordinate points into a continuous gradient. This distributes the density values outward, creating the visual "glowing hot spot" effect.

#### Q13: What does the "Gaussian Kernel Size" mean?
**Answer:** The kernel size (e.g., 35x35) defines the size of the pixel neighborhood used to calculate the blurred value of a pixel. A larger kernel size produces a wider, smoother heat spread, while a smaller kernel size creates tighter, more localized hot spots. The kernel size must always be an odd number.

#### Q14: How does OpenCV map grayscale density to a colored heatmap?
**Answer:** We normalize the accumulated float array to the standard grayscale range of 0-255. Then, we pass it to `cv2.applyColorMap(blurred, cv2.COLORMAP_JET)`. The Jet colormap maps low values (near 0) to blue (cold), middle values to yellow/green, and high values (near 255) to red (hot).

#### Q15: Explain OpenCV's Alpha Blending equation used in your project.
**Answer:** To show the heatmap overlaid on top of the store floor, we blend the two images using:
$$\text{Output} = (1 - \alpha) \cdot \text{BaseFrame} + \alpha \cdot \text{ColormappedHeatmap}$$
Where $\alpha$ (alpha) is the transparency factor (e.g., 0.55). We perform this operation selectively using a mask so that areas with zero heat remain completely transparent and do not get colored blue.

---

### Part 4: Dashboard & Analytics Questions

#### Q16: Why did you choose Streamlit for the dashboard?
**Answer:** Streamlit allows us to build interactive web applications in pure Python without writing HTML, CSS, or JavaScript. It integrates easily with scientific libraries like OpenCV, Pandas, Matplotlib, and Seaborn, making it ideal for prototyping and presenting data-heavy AI projects.

#### Q17: What is Streamlit's Session State and why is it used?
**Answer:** By default, Streamlit re-runs the entire python script from top to bottom every time a widget changes. This resets all local variables. We use `st.session_state` to store our `CCTVTracker`, `HeatmapAccumulator`, and timeline history so that their data persists across script re-runs and does not get wiped out during interaction.

#### Q18: How do you calculate customer dwell time in a specific store zone?
**Answer:** We define zones using coordinates `(x1, y1, x2, y2)`. In each frame, if a customer's center point is within these boundaries, we increment a frame counter for that customer in that zone. The total dwell time in seconds is:
$$\text{Dwell Time} = \frac{\text{Accumulated Frames}}{\text{Video Frame Rate (FPS)}}$$

#### Q19: How are the analytics charts populated?
**Answer:** We compile the tracker data into a Pandas DataFrame. We use:
1. **Seaborn and Matplotlib** to plot a line graph of the customer count over time.
2. A **horizontal bar chart** of cumulative dwell times across different zones to show which sections of the store attracted the most customer attention.
These plots are displayed in Streamlit using `st.pyplot(fig)`.

---

### Part 5: Project Design & Engineering Questions

#### Q20: What is the purpose of the Synthetic Video Generator?
**Answer:** It serves two critical purposes:
1. **Instant Portability**: Downloading large CCTV videos can fail due to networks or disk limits. The generator creates a local video file immediately.
2. **Deterministic Evaluation**: It generates a companion CSV file containing ground-truth tracking paths. This allows the system to run in a fast "Emulation Mode" that bypasses YOLOv8 inference, allowing the dashboard to run smoothly at 30 FPS on laptops without GPUs.

#### Q21: What are the main limitations of this system?
**Answer:**
1. **Occlusion**: If a customer is completely hidden behind another person or a high shelf, their tracking ID might break and reset.
2. **Camera Angle**: Top-down angles are best for tracking but make facial recognition or classification difficult, whereas oblique angles introduce perspective distortion.
3. **Lighting Changes**: Extreme shadows or dark store corners can reduce YOLOv8's detection confidence.

#### Q22: How can this system be optimized to run faster in a real store?
**Answer:**
1. **Model Quantization**: Export the PyTorch YOLO model to ONNX, OpenVINO, or TensorRT formats to run faster on Intel or NVIDIA chips.
2. **Frame Skipping**: Run the YOLO detector on every 3rd or 5th frame instead of every frame, and use cheaper tracker predictions (like Kalman Filter tracking) for intermediate frames.
3. **Resolution Downscaling**: Resize high-definition input frames (like 4K) to 640x640 before feeding them to YOLO.

#### Q23: How would you scale this to a multi-camera store environment?
**Answer:** We would implement **Multi-Camera Multi-Target Tracking (MCMTT)**. This involves extracting appearance feature vectors (embeddings) for each shopper using a Re-Identification (Re-ID) network. When a shopper leaves camera 1's view and enters camera 2's view, we compare their embedding to active tracks using cosine similarity to maintain the same ID.

#### Q24: What security/privacy implications exist?
**Answer:** Under privacy frameworks like GDPR, collecting identifiable biometric data is restricted. Our system is privacy-friendly by design because:
1. We only track generic bounding box coordinates.
2. We do not run facial recognition or store individual identities.
3. The tracking coordinates are aggregated into anonymous statistical metrics.

#### Q25: How does this system differentiate between store employees and customers?
**Answer:** Our project implements an active **Computer Vision Uniform Classification & Role Isolation Pipeline**:
1. **Torso Region Segmentation**: For each detected person bounding box $[x_1, y_1, x_2, y_2]$, the algorithm extracts the upper torso crop $[y_1 + 0.15h : y_1 + 0.60h, x_1 + 0.20w : x_2 - 0.20w]$, which isolates the shirt/vest while avoiding hair, face, legs, and background margins.
2. **HSV Color Space Masking**: The torso crop is converted to Hue-Saturation-Value (HSV). HSV decouples chromatic hue from brightness, making color matching resilient to shadows and overhead store lighting. We apply threshold masks against preset uniform profiles (e.g., Red for Target/Coles, Blue for Walmart, Green/Neon for Safety Vests, Black, or Custom Hex).
3. **Temporal Identity Smoothing**: To prevent classification flickering due to sudden shadows or turns, each track maintains a rolling window of classification votes. If $\ge 35\%$ of recent frames match the uniform, the track is locked as `STAFF`.
4. **Heatmap & Metric Isolation**: Staff coordinates are completely excluded from the heatmap accumulator (preventing false hot-spots) and excluded from customer footfall and dwell calculations. Staff are visually labeled with a distinct amber badge `👔 STAFF #[ID] (Filtered)` on the CCTV feed for transparency.

---

### Part 6: Staff Detection & Analytics Isolation Questions

#### Q26: Why is converting RGB to HSV essential for uniform color detection?
**Answer:** In RGB color space, chromatic color and luminance (brightness) are entangled. If an employee walks from a bright fluorescent light into a shadowy aisle corner, their RGB pixel values change drastically. In HSV (Hue, Saturation, Value), the **Hue** represents the pure color wavelength (e.g., Red = 0-10 or 170-180, Blue = 95-130) regardless of brightness. The **Value** and **Saturation** can fluctuate with store lighting while the Hue remains stable, ensuring consistent uniform detection.

#### Q27: Why must staff be excluded from retail heatmap analytics?
**Answer:** Store employees spend long periods restocking shelves, standing at cashier registers, or walking aisles repeatedly throughout their shifts. If tracked as customers:
1. They generate artificial **hot-spots** on the heatmap in areas where customers rarely dwell.
2. They inflate **total footfall** counts and distort average customer dwell time calculations.
3. They bias **aisle popularity metrics**, misleading store managers when planning inventory placement and promotions.

#### Q28: How does the system handle temporary occlusion or partial turning of a staff member?
**Answer:** The system uses **Temporal Voting with Hysteresis**:
- A rolling buffer of the last 15 frame classifications is maintained per `track_id`.
- If a staff member turns their back or is partially occluded for a few frames, their recent history preserves the `STAFF` classification.
- An identity change from `STAFF` to `CUSTOMER` requires multiple consecutive non-matching frames, eliminating visual flicker.
