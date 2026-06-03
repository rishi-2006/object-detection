import time
from datetime import datetime
from collections import deque
import cv2
import pandas as pd
import numpy as np

class AnalyticsManager:
    def __init__(self, max_path_length=30):
        """
        Initializes the Analytics Manager.
        
        Args:
            max_path_length: Maximum number of points to keep in the motion trail history.
        """
        self.max_path_length = max_path_length
        self.reset()

    def reset(self):
        """
        Resets all analytics data for a new session.
        """
        self.start_time = time.time()
        self.fps_history = deque(maxlen=100)
        self.active_counts = {}       # Current frame object counts: {class_name: count}
        self.cumulative_ids = {}      # Track IDs seen so far: {class_name: set(track_ids)}
        self.detection_history = []   # List of logs: {'timestamp', 'track_id', 'class_name', 'confidence'}
        self.track_paths = {}         # Trajectory coordinates: {track_id: deque(maxlen=max_path_length)}
        self.confidences = []         # List of all detection confidences
        self.logged_ids = set()       # Set of track IDs already added to history logs

    def update(self, active_tracks, current_fps):
        """
        Updates the analytics state for the current frame.
        
        Args:
            active_tracks: List of active track dictionaries from tracker.py.
            current_fps: Calculated FPS for the current frame.
        """
        # Save FPS
        if current_fps > 0:
            self.fps_history.append(current_fps)
            
        # Reset active counts for this frame
        self.active_counts = {}
        
        # Track IDs currently visible to clean up inactive paths later
        current_frame_ids = set()
        
        for track in active_tracks:
            track_id = track["track_id"]
            class_name = track["class_name"]
            conf = track["confidence"]
            bbox = track["bbox"]
            
            current_frame_ids.add(track_id)
            
            # 1. Update Active Counts
            self.active_counts[class_name] = self.active_counts.get(class_name, 0) + 1
            
            # 2. Update Cumulative Counts
            if class_name not in self.cumulative_ids:
                self.cumulative_ids[class_name] = set()
            self.cumulative_ids[class_name].add(track_id)
            
            # 3. Save confidence scores
            self.confidences.append(conf)
            
            # 4. Calculate bbox center
            left, top, right, bottom = bbox
            center_x = int((left + right) / 2)
            center_y = int((top + bottom) / 2)
            
            # 5. Update Movement Trajectory
            if track_id not in self.track_paths:
                self.track_paths[track_id] = deque(maxlen=self.max_path_length)
            self.track_paths[track_id].append((center_x, center_y))
            
            # 6. Log New Detections (only once per unique Track ID to avoid flooding logs)
            if track_id not in self.logged_ids:
                self.logged_ids.add(track_id)
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
                self.detection_history.append({
                    "Timestamp": timestamp,
                    "Track ID": track_id,
                    "Class": class_name.capitalize(),
                    "Confidence": f"{conf:.2%}"
                })
                
        # Clean up paths for track IDs that have been lost/deleted (not in current frame)
        # Note: We keep them for a brief period or let DeepSORT's max_age handle deletion.
        # To avoid path trails popping out immediately, we can delete them if their tracker is gone.
        lost_tracks = set(self.track_paths.keys()) - current_frame_ids
        # If the number of paths is very large, clean up lost tracks
        for lost_id in lost_tracks:
            # We can optionally keep them but delete if we exceed 100 inactive tracks
            if len(self.track_paths) > 200:
                del self.track_paths[lost_id]

    def draw_paths(self, frame, active_tracks, trail_color=(0, 243, 255), thickness=2):
        """
        Draws motion trails on the frame for active tracked objects.
        
        Args:
            frame: OpenCV image frame.
            active_tracks: List of active tracks.
            trail_color: Color tuple (B, G, R) for the trail.
            thickness: Width of the drawn line.
        """
        for track in active_tracks:
            track_id = track["track_id"]
            if track_id in self.track_paths:
                points = list(self.track_paths[track_id])
                
                # Draw lines connecting sequential center points
                for i in range(1, len(points)):
                    if points[i - 1] is None or points[i] is None:
                        continue
                        
                    # Calculate fading alpha for older trail points
                    alpha = float(i) / len(points)
                    # Create fading color
                    b = int(trail_color[0] * alpha)
                    g = int(trail_color[1] * alpha)
                    r = int(trail_color[2] * alpha)
                    
                    cv2.line(frame, points[i - 1], points[i], (b, g, r), int(thickness * alpha + 1))
                    
                # Draw a neon dot at the current location
                if points:
                    cv2.circle(frame, points[-1], 4, (0, 0, 255), -1) # Red core
                    cv2.circle(frame, points[-1], 6, trail_color, 1)  # Glowing outline

    def get_summary_stats(self):
        """
        Calculates and returns current session metrics.
        """
        avg_fps = np.mean(self.fps_history) if self.fps_history else 0.0
        avg_conf = np.mean(self.confidences) if self.confidences else 0.0
        
        # Get cumulative count by class
        cumulative_counts = {cls: len(ids) for cls, ids in self.cumulative_ids.items()}
        total_unique_objects = sum(cumulative_counts.values())
        
        duration = time.time() - self.start_time
        
        return {
            "duration_seconds": round(duration, 1),
            "average_fps": round(float(avg_fps), 1),
            "average_confidence": round(float(avg_conf), 4),
            "total_objects_tracked": total_unique_objects,
            "cumulative_counts": cumulative_counts,
            "active_counts": self.active_counts
        }

    def export_csv(self):
        """
        Exports the detection history logs as a CSV formatted string.
        """
        if not self.detection_history:
            return ""
        df = pd.DataFrame(self.detection_history)
        return df.to_csv(index=False)
