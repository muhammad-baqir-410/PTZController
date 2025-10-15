#!/usr/bin/env python3
"""
Person Tracking Camera Controller
Uses Ultralytics YOLO for person detection and automatic camera following
"""

import cv2
import numpy as np
import sys
import os
import threading
import time
from typing import Optional, Tuple, List
from collections import deque

# Add the PTZController module to the path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'PTZController'))

from PTZController.camera import Camera
from PTZController.config import Config
from PTZController import logger

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("Warning: Ultralytics not available. Install with: pip install ultralytics")


class PersonTracker:
    """Person tracking using YOLO - optimized for real-time performance."""
    
    def __init__(self, model_name='yolov8n.pt'):
        """Initialize the person tracker."""
        if not YOLO_AVAILABLE:
            raise ImportError("Ultralytics not available. Install with: pip install ultralytics")
        
        print(f"Loading YOLO model: {model_name}")
        self.model = YOLO(model_name)
        self.person_class_id = 0  # Person class ID in COCO dataset
        
        # Tracking parameters
        self.tracking_history = deque(maxlen=5)  # Keep last 5 detections
        self.last_person_center = None
        self.tracking_lost_frames = 0
        self.max_lost_frames = 15  # Stop tracking after 15 frames without person
        
        # Performance optimization
        self.inference_interval = 3  # Run YOLO every 3 frames
        self.frame_count = 0
        self.last_detection = None
        self.prediction_confidence = 0.3  # Lower confidence for faster detection
        
    def detect_person(self, frame):
        """Detect person in frame and return center coordinates - optimized for speed."""
        # Use smaller input size for faster inference
        results = self.model(frame, imgsz=416, verbose=False, conf=self.prediction_confidence)
        
        person_centers = []
        confidences = []
        
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    # Check if it's a person (class 0) with confidence threshold
                    if int(box.cls[0]) == self.person_class_id and float(box.conf[0]) > self.prediction_confidence:
                        # Get bounding box coordinates
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        
                        # Calculate center
                        center_x = int((x1 + x2) / 2)
                        center_y = int((y1 + y2) / 2)
                        
                        person_centers.append((center_x, center_y))
                        confidences.append(float(box.conf[0]))
        
        # Return the person with highest confidence
        if person_centers:
            best_idx = np.argmax(confidences)
            return person_centers[best_idx], confidences[best_idx]
        
        return None, 0.0
    
    def get_tracking_target(self, frame):
        """Get the target position for camera tracking - optimized for real-time."""
        self.frame_count += 1
        
        # Only run YOLO inference every few frames for performance
        if self.frame_count % self.inference_interval == 0:
            person_center, confidence = self.detect_person(frame)
            self.last_detection = (person_center, confidence)
        else:
            # Use last detection for intermediate frames
            person_center, confidence = self.last_detection if self.last_detection else (None, 0.0)
        
        if person_center is not None:
            self.last_person_center = person_center
            self.tracking_lost_frames = 0
            self.tracking_history.append(person_center)
            return person_center, confidence
        else:
            self.tracking_lost_frames += 1
            return None, 0.0


class AutoFollowCamera:
    """Camera controller with automatic person following."""
    
    def __init__(self, config_file: str = "PTZController.conf", camera_id: int = 1):
        """Initialize the auto-follow camera."""
        self.config = Config(config_file)
        self.camera_id = camera_id
        self.current_camera = None
        self.stream_reader = None
        self.tracker = None
        self.running = False
        
        # Control parameters
        self.pan_speed = 0.3
        self.tilt_speed = 0.3
        self.zoom_speed = 0.3
        
        # Tracking parameters
        self.target_center_x = 0.5  # Target center position (0.0 to 1.0)
        self.target_center_y = 0.5
        self.dead_zone = 0.15  # Dead zone around center (15% of frame) - larger for stability
        self.smooth_factor = 0.2  # Smoothing factor for movements - increased for smoother motion
        
        # Performance optimization
        self.last_movement_time = 0
        self.movement_interval = 0.1  # Minimum time between movements (100ms)
        
        # Initialize camera and tracker
        self.initialize_camera()
        self.initialize_tracker()
        
    def initialize_camera(self):
        """Initialize the camera."""
        print(f"Initializing camera {self.camera_id}...")
        
        # Get camera configuration
        camera_sections = [s for s in self.config.sections() if s.startswith('camera') or s.startswith('Camera')]
        if self.camera_id > len(camera_sections):
            raise ValueError(f"Camera {self.camera_id} not found in configuration")
        
        section = camera_sections[self.camera_id - 1]
        camera_options = {}
        camera_options['id'] = self.camera_id
        for key, value in self.config.items(section):
            if key == 'id':
                continue
            camera_options[key] = value
        if 'name' not in camera_options:
            camera_options['name'] = section[:12] if len(section) > 12 else section
            
        self.current_camera = Camera(camera_options)
        
        # Wait for camera to connect
        print("Waiting for camera to connect...")
        time.sleep(3)
        
        if self.current_camera.isconnected:
            print(f"✅ Connected to camera: {self.current_camera.name}")
        else:
            raise ConnectionError("Camera not connected!")
    
    def initialize_tracker(self):
        """Initialize the person tracker."""
        print("Initializing person tracker...")
        try:
            self.tracker = PersonTracker()
            print("✅ Person tracker ready")
        except Exception as e:
            print(f"❌ Failed to initialize tracker: {e}")
            raise
    
    def get_stream_uri(self) -> str:
        """Get the RTSP stream URI."""
        username = self.current_camera._Camera__userid
        password = self.current_camera._Camera__password
        host = self.current_camera.host
        return f"rtsp://{username}:{password}@{host}:554/Stream/Live/101"
    
    def start_stream(self):
        """Start the video stream."""
        print("Starting video stream...")
        
        stream_uri = self.get_stream_uri()
        
        try:
            self.cap = cv2.VideoCapture(stream_uri, cv2.CAP_FFMPEG)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            self.cap.set(cv2.CAP_PROP_FPS, 60)
            
            if not self.cap.isOpened():
                raise RuntimeError("Failed to open video stream")
            
            print("✅ Video stream started")
            return True
            
        except Exception as e:
            print(f"❌ Error starting stream: {e}")
            return False
    
    def calculate_movement(self, person_center, frame_width, frame_height):
        """Calculate camera movement needed to center the person."""
        if person_center is None:
            return 0, 0
        
        person_x, person_y = person_center
        
        # Convert to relative coordinates (0.0 to 1.0)
        rel_x = person_x / frame_width
        rel_y = person_y / frame_height
        
        # Calculate error from target center
        error_x = rel_x - self.target_center_x
        error_y = rel_y - self.target_center_y
        
        # Check if within dead zone
        if abs(error_x) < self.dead_zone and abs(error_y) < self.dead_zone:
            return 0, 0
        
        # Calculate movement speeds (proportional control)
        pan_speed = error_x * self.pan_speed
        tilt_speed = -error_y * self.tilt_speed  # Negative because camera coordinates are inverted
        
        # Apply smoothing
        pan_speed *= self.smooth_factor
        tilt_speed *= self.smooth_factor
        
        # Clamp to maximum speeds
        pan_speed = max(-self.pan_speed, min(self.pan_speed, pan_speed))
        tilt_speed = max(-self.tilt_speed, min(self.tilt_speed, tilt_speed))
        
        return pan_speed, tilt_speed
    
    def move_camera(self, pan_speed, tilt_speed):
        """Move the camera with given speeds - optimized for real-time."""
        current_time = time.time()
        
        # Throttle movements to avoid overwhelming the camera
        if current_time - self.last_movement_time < self.movement_interval:
            return
            
        if abs(pan_speed) > 0.01 or abs(tilt_speed) > 0.01:
            try:
                self.current_camera.move_continuous((pan_speed, tilt_speed, 0))
                self.last_movement_time = current_time
            except Exception as e:
                print(f"Camera movement error: {e}")
        else:
            # Stop camera if no significant movement needed
            try:
                self.current_camera.stop()
            except:
                pass
    
    def run_auto_follow(self):
        """Run the automatic person following."""
        if not self.start_stream():
            return
        
        print("\n🎯 Auto-follow mode started!")
        print("The camera will automatically follow detected persons")
        print("Press ESC to exit, SPACE to toggle tracking")
        
        tracking_enabled = True
        frame_count = 0
        last_info_time = time.time()
        last_fps_time = time.time()
        fps_counter = 0
        
        # Create display window
        cv2.namedWindow('Person Tracking Camera', cv2.WINDOW_AUTOSIZE)
        
        try:
            while self.running:
                ret, frame = self.cap.read()
                if not ret:
                    print("Failed to read frame")
                    break
                
                frame_count += 1
                fps_counter += 1
                height, width = frame.shape[:2]
                
                # Detect person if tracking is enabled
                person_center = None
                confidence = 0.0
                
                if tracking_enabled:
                    person_center, confidence = self.tracker.get_tracking_target(frame)
                    
                    if person_center is not None:
                        # Calculate and execute camera movement
                        pan_speed, tilt_speed = self.calculate_movement(person_center, width, height)
                        self.move_camera(pan_speed, tilt_speed)
                        
                        # Draw tracking info
                        cv2.circle(frame, person_center, 10, (0, 255, 0), 2)
                        cv2.putText(frame, f"Person: {confidence:.2f}", (10, 30), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                        cv2.putText(frame, "TRACKING", (10, 60), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    else:
                        # No person detected
                        if self.tracker.tracking_lost_frames > self.tracker.max_lost_frames:
                            self.move_camera(0, 0)  # Stop camera
                        
                        cv2.putText(frame, "NO PERSON DETECTED", (10, 30), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                        cv2.putText(frame, f"Lost: {self.tracker.tracking_lost_frames}", (10, 60), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                else:
                    cv2.putText(frame, "TRACKING DISABLED", (10, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
                
                # Draw center crosshair
                center_x = int(width * self.target_center_x)
                center_y = int(height * self.target_center_y)
                cv2.line(frame, (center_x - 20, center_y), (center_x + 20, center_y), (255, 255, 255), 2)
                cv2.line(frame, (center_x, center_y - 20), (center_x, center_y + 20), (255, 255, 255), 2)
                
                # Calculate and display FPS
                current_time = time.time()
                if current_time - last_fps_time >= 1.0:  # Every second
                    fps = fps_counter / (current_time - last_fps_time)
                    fps_counter = 0
                    last_fps_time = current_time
                else:
                    fps = 0
                
                # Add status info
                if fps > 0:
                    cv2.putText(frame, f"FPS: {fps:.1f}", (10, height - 80), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                cv2.putText(frame, f"Frame: {frame_count}", (10, height - 60), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                cv2.putText(frame, "ESC=Exit, SPACE=Toggle", (10, height - 40), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                cv2.putText(frame, f"Camera: {self.current_camera.name}", (10, height - 20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                # Display frame
                cv2.imshow('Person Tracking Camera', frame)
                
                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                if key == 27:  # ESC
                    break
                elif key == ord(' '):  # SPACE
                    tracking_enabled = not tracking_enabled
                    print(f"Tracking {'enabled' if tracking_enabled else 'disabled'}")
                elif key == ord('h'):  # H - Go home
                    self.current_camera.go_home()
                    print("Camera moved to home position")
                elif key == ord('s'):  # S - Stop
                    self.current_camera.stop()
                    print("Camera stopped")
                
                # Reduced periodic info to avoid console spam
                current_time = time.time()
                if current_time - last_info_time >= 10:  # Every 10 seconds
                    if person_center:
                        print(f"Tracking person at ({person_center[0]}, {person_center[1]}) conf:{confidence:.2f} fps:{fps:.1f}")
                    else:
                        print(f"No person detected - fps:{fps:.1f}")
                    last_info_time = current_time
                    
        except KeyboardInterrupt:
            print("\nShutting down...")
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Cleanup resources."""
        self.running = False
        if hasattr(self, 'cap'):
            self.cap.release()
        cv2.destroyAllWindows()
        
        # Stop camera movement
        if self.current_camera and self.current_camera.isconnected:
            try:
                self.current_camera.stop()
            except:
                pass
    
    def run(self):
        """Main run method."""
        print("PERSON TRACKING CAMERA CONTROLLER")
        print("=================================")
        
        if not YOLO_AVAILABLE:
            print("❌ Ultralytics not available. Install with: pip install ultralytics")
            return
        
        self.running = True
        self.run_auto_follow()


def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Person Tracking Camera Controller')
    parser.add_argument('--camera', type=int, default=1, help='Camera ID to use (default: 1)')
    parser.add_argument('--config', type=str, default='PTZController.conf', help='Configuration file')
    
    args = parser.parse_args()
    
    try:
        controller = AutoFollowCamera(args.config, args.camera)
        controller.run()
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
