import cv2
import numpy as np

class HeatmapAccumulator:
    def __init__(self, width=1280, height=720, decay=1.0):
        """
        Initializes the heatmap accumulator.
        decay: multiplier for values on each update if we want temporary heat maps (not used for cumulative).
        """
        self.width = width
        self.height = height
        self.decay = decay
        self.reset()
        
    def reset(self):
        """Resets the accumulated density map to zero."""
        self.accum_map = np.zeros((self.height, self.width), dtype=np.float32)
        
    def update(self, centers):
        """
        Updates the heatmap with new coordinates.
        centers: List of tuples (x, y) of tracked customer centers.
        """
        # Apply decay if we want historical heat to fade (for rolling heatmaps)
        if self.decay < 1.0:
            self.accum_map *= self.decay
            
        for x, y in centers:
            # Check bounds to prevent index errors
            if 0 <= x < self.width and 0 <= y < self.height:
                # Increment a small region around the center to create a wider footprint
                # Drawing a small circle on a temporary mask is faster and cleaner
                cv2.circle(self.accum_map, (int(x), int(y)), radius=15, color=1.0, thickness=-1)
                
    def get_heatmap(self, base_frame, alpha=0.5, blur_strength=31, colormap=cv2.COLORMAP_JET):
        """
        Generates and blends the heatmap overlay onto the base frame.
        base_frame: The background image/video frame.
        alpha: Blending transparency (0 = only video, 1 = only heatmap).
        blur_strength: Size of Gaussian blur kernel (must be odd).
        colormap: OpenCV colormap code (e.g., cv2.COLORMAP_JET, cv2.COLORMAP_HOT).
        """
        if np.max(self.accum_map) == 0:
            # If no activity yet, return original frame and blank heatmap
            blank_heatmap = np.zeros_like(base_frame)
            return base_frame.copy(), blank_heatmap
            
        # 1. Normalize the accumulator map to 0-255 range
        normalized = np.zeros_like(self.accum_map, dtype=np.uint8)
        cv2.normalize(self.accum_map, normalized, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        
        # 2. Apply Gaussian Blur to smooth the hot spots
        if blur_strength % 2 == 0:
            blur_strength += 1  # Kernel size must be odd
        blurred = cv2.GaussianBlur(normalized, (blur_strength, blur_strength), 0)
        
        # 3. Apply color map (converts single-channel gray density to 3-channel RGB/BGR)
        heatmap_color = cv2.applyColorMap(blurred, colormap)
        
        # 4. Create a mask to prevent drawing heatmap where there is no heat
        # Threshold the blurred map so that cold zones (0 heat) remain transparent
        _, threshold_mask = cv2.threshold(blurred, 10, 255, cv2.THRESH_BINARY)
        
        # Mask the heatmap color representation
        heatmap_masked = cv2.bitwise_and(heatmap_color, heatmap_color, mask=threshold_mask)
        
        # 5. Blend the masked heatmap color with the base frame
        blended_frame = base_frame.copy()
        
        # Apply the heatmap only where we have activity (threshold_mask > 0)
        indices = threshold_mask > 0
        blended_frame[indices] = cv2.addWeighted(
            base_frame, 1.0 - alpha, heatmap_color, alpha, 0
        )[indices]
        
        return blended_frame, heatmap_masked
