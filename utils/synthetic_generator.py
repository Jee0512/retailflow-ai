import os
import cv2
import numpy as np
import pandas as pd
import random

def generate_video(output_path="sample_cctv.mp4", num_frames=900, width=1280, height=720):
    """
    Generates a synthetic retail CCTV video and a matching detections CSV.
    The video simulates shoppers walking around a store floor plan, as well as
    a store staff associate in uniform patrolling the aisles.
    """
    csv_path = output_path.replace(".mp4", "_detections.csv")
    
    # Define store layout zones (centers and bounding boxes)
    zones = {
        "Entrance": {"rect": (50, 500, 150, 680), "color": (46, 204, 113)},      # Green
        "Aisle A (Produce)": {"rect": (150, 100, 450, 250), "color": (241, 196, 15)},# Yellow
        "Aisle B (Apparel)": {"rect": (550, 100, 850, 250), "color": (52, 152, 219)},# Blue
        "Aisle C (Electronics)": {"rect": (350, 320, 750, 470), "color": (155, 89, 182)},# Purple
        "Checkout 1": {"rect": (900, 500, 1050, 650), "color": (231, 76, 60)},     # Red
        "Checkout 2": {"rect": (1100, 500, 1250, 650), "color": (230, 126, 34)},    # Orange
        "Exit": {"rect": (50, 300, 150, 450), "color": (149, 165, 166)}            # Gray
    }
    
    # Store background initialization
    def draw_background():
        bg = np.zeros((height, width, 3), dtype=np.uint8)
        # Fill floor with a modern dark gray
        bg[:] = (26, 26, 26)
        
        # Draw grid lines for high-tech look
        for x in range(0, width, 80):
            cv2.line(bg, (x, 0), (x, height), (35, 35, 35), 1)
        for y in range(0, height, 80):
            cv2.line(bg, (0, y), (width, y), (35, 35, 35), 1)
            
        # Draw store boundary walls
        cv2.rectangle(bg, (10, 10), (width - 10, height - 10), (80, 80, 80), 3)
        
        # Draw zone boxes and labels
        for name, info in zones.items():
            x1, y1, x2, y2 = info["rect"]
            color = info["color"]
            
            # Semi-transparent zone backing
            overlay = bg.copy()
            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
            cv2.addWeighted(overlay, 0.15, bg, 0.85, 0, bg)
            
            # Zone border
            cv2.rectangle(bg, (x1, y1), (x2, y2), color, 2)
            
            # Label background
            lbl_w = len(name) * 8 + 10
            cv2.rectangle(bg, (x1 + 5, y1 - 25), (x1 + 5 + lbl_w, y1 - 2), (40, 40, 40), -1)
            cv2.rectangle(bg, (x1 + 5, y1 - 25), (x1 + 5 + lbl_w, y1 - 2), color, 1)
            cv2.putText(bg, name, (x1 + 10, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (240, 240, 240), 1, cv2.LINE_AA)
            
        return bg

    # Pre-render layout base image
    base_bg = draw_background()
    
    # Video writer setup
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, 30.0, (width, height))
    
    # Active shopper agent states
    shoppers = {}
    next_shopper_id = 1
    
    # Persistent Staff Associate (ID 99) in Red uniform patrolling aisles and checkout
    staff_waypoints = [
        [300, 180], # Aisle A
        [700, 180], # Aisle B
        [550, 400], # Aisle C
        [980, 580], # Checkout 1
        [550, 400], # Return to Aisle C
    ]
    staff_member = {
        "id": 99,
        "pos": [300.0, 180.0],
        "targets": staff_waypoints,
        "target_idx": 0,
        "dwell_timer": 30,
        "color": (20, 20, 220), # Vibrant BGR Red uniform
        "speed": 2.2,
        "width": 42,
        "height": 82,
        "role": "staff"
    }
    
    detection_records = []
    
    # Helper to get random point in a zone rect
    def get_random_point_in_zone(zone_name):
        x1, y1, x2, y2 = zones[zone_name]["rect"]
        return [random.randint(x1 + 25, x2 - 25), random.randint(y1 + 25, y2 - 25)]
        
    for frame_idx in range(num_frames):
        # Spawning logic: spawn a shopper with a certain probability
        if len(shoppers) < 8 and random.random() < 0.05:
            possible_aisles = ["Aisle A (Produce)", "Aisle B (Apparel)", "Aisle C (Electronics)"]
            selected_aisles = random.sample(possible_aisles, k=random.randint(1, 3))
            checkout = random.choice(["Checkout 1", "Checkout 2"])
            
            target_names = ["Entrance"] + selected_aisles + [checkout, "Exit"]
            target_coords = [get_random_point_in_zone(t) for t in target_names]
            
            # Non-uniform random shopper clothing colors (excluding bright red)
            shopper_color = (random.randint(110, 240), random.randint(110, 240), random.randint(30, 140))
            
            shoppers[next_shopper_id] = {
                "pos": list(target_coords[0]),
                "targets": target_coords,
                "target_names": target_names,
                "current_target_idx": 1,
                "dwell_timer": 0,
                "color": shopper_color,
                "speed": random.uniform(2.5, 4.5),
                "width": 40,
                "height": 80,
                "role": "customer"
            }
            next_shopper_id += 1
            
        # Draw frame
        frame = base_bg.copy()
        shoppers_to_remove = []
        
        # 1. Update & draw shoppers
        for s_id, s in shoppers.items():
            curr_pos = s["pos"]
            target_idx = s["current_target_idx"]
            
            if target_idx < len(s["targets"]):
                target_pos = s["targets"][target_idx]
                target_name = s["target_names"][target_idx]
                
                dx = target_pos[0] - curr_pos[0]
                dy = target_pos[1] - curr_pos[1]
                dist = np.hypot(dx, dy)
                
                if dist > 8:
                    noise_x = random.uniform(-1.0, 1.0)
                    noise_y = random.uniform(-1.0, 1.0)
                    s["pos"][0] += (dx / dist) * s["speed"] + noise_x
                    s["pos"][1] += (dy / dist) * s["speed"] + noise_y
                else:
                    if s["dwell_timer"] == 0:
                        if target_name in ["Entrance", "Exit"]:
                            s["dwell_timer"] = 1
                        elif "Checkout" in target_name:
                            s["dwell_timer"] = random.randint(60, 120)
                        else:
                            s["dwell_timer"] = random.randint(40, 100)
                            
                    s["dwell_timer"] -= 1
                    if s["dwell_timer"] <= 0:
                        s["current_target_idx"] += 1
            else:
                shoppers_to_remove.append(s_id)
                continue
                
            px, py = int(s["pos"][0]), int(s["pos"][1])
            sw, sh = s["width"], s["height"]
            
            x1 = int(px - sw/2)
            y1 = int(py - sh/2)
            x2 = int(px + sw/2)
            y2 = int(py + sh/2)
            
            detection_records.append({
                "frame_idx": frame_idx,
                "track_id": s_id,
                "x_min": x1,
                "y_min": y1,
                "x_max": x2,
                "y_max": y2,
                "role": "customer"
            })
            
            # Shopper avatar (head + torso oval)
            cv2.ellipse(frame, (px, py + 10), (14, 25), 0, 0, 360, s["color"], -1)
            cv2.circle(frame, (px, py - 18), 8, s["color"], -1)
            cv2.circle(frame, (px, py - 18), 4, (255, 255, 255), -1)
            cv2.putText(frame, f"Shopper {s_id}", (x1 - 5, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

        for s_id in shoppers_to_remove:
            shoppers.pop(s_id, None)
            
        # 2. Update & draw staff member (ID 99)
        st_pos = staff_member["pos"]
        st_target = staff_member["targets"][staff_member["target_idx"]]
        st_dx = st_target[0] - st_pos[0]
        st_dy = st_target[1] - st_pos[1]
        st_dist = np.hypot(st_dx, st_dy)
        
        if st_dist > 6:
            st_pos[0] += (st_dx / st_dist) * staff_member["speed"]
            st_pos[1] += (st_dy / st_dist) * staff_member["speed"]
        else:
            staff_member["dwell_timer"] -= 1
            if staff_member["dwell_timer"] <= 0:
                staff_member["target_idx"] = (staff_member["target_idx"] + 1) % len(staff_member["targets"])
                staff_member["dwell_timer"] = random.randint(45, 90) # Patrol inspection dwell
                
        st_px, st_py = int(st_pos[0]), int(st_pos[1])
        st_w, st_h = staff_member["width"], staff_member["height"]
        st_x1, st_y1 = int(st_px - st_w/2), int(st_py - st_h/2)
        st_x2, st_y2 = int(st_px + st_w/2), int(st_py + st_h/2)
        
        detection_records.append({
            "frame_idx": frame_idx,
            "track_id": staff_member["id"],
            "x_min": st_x1,
            "y_min": st_y1,
            "x_max": st_x2,
            "y_max": st_y2,
            "role": "staff"
        })
        
        # Draw Staff avatar with clear Red uniform vest & name tag
        cv2.ellipse(frame, (st_px, st_py + 10), (15, 26), 0, 0, 360, staff_member["color"], -1) # Red uniform body
        cv2.circle(frame, (st_px, st_py - 18), 8, (230, 230, 230), -1) # Head
        cv2.rectangle(frame, (st_px - 8, st_py + 4), (st_px + 8, st_py + 12), (255, 255, 255), -1) # Name badge
        cv2.putText(frame, "Staff 99", (st_x1 - 5, st_y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 165, 255), 1, cv2.LINE_AA)

        # 3. Draw CCTV Overlay HUD details
        cv2.putText(frame, "CCTV CAM_01 - RETAIL STORE MAIN FLOOR", (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (240, 240, 240), 2, cv2.LINE_AA)
        
        seconds = frame_idx // 30
        mins = (seconds // 60)
        secs = seconds % 60
        timestamp_str = f"2026-05-28 13:02:{secs:02d}" if mins == 0 else f"2026-05-28 13:{mins+2:02d}:{secs:02d}"
        cv2.putText(frame, timestamp_str, (width - 260, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, "REC [O]", (width - 110, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2, cv2.LINE_AA)
        
        out.write(frame)
        
    out.release()
    
    df = pd.DataFrame(detection_records)
    df.to_csv(csv_path, index=False)
    
    print(f"Generated {num_frames} frames to {output_path}")
    print(f"Generated bounding box tracking data to {csv_path} with staff associate ID 99")

if __name__ == "__main__":
    generate_video()
