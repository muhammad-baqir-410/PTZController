#!/usr/bin/env python3
"""
Threaded Person Tracking Camera Controller
Uses separate threads for frame capture, display, and tracking
Uses Ultralytics track() for efficient person tracking
"""

import cv2
import numpy as np
import sys
import os
import threading
import time
import queue
from typing import Optional, Tuple, List
from collections import deque
from datetime import datetime

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

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("Warning: PyTorch not available for TensorRT optimization")


class ThreadedVideoCapture:
    """Threaded video capture for real-time frame reading."""
    
    def __init__(self, rtsp_url):
        self.rtsp_url = rtsp_url
        self.cap = None
        self.frame = None
        self.lock = threading.Lock()
        self.running = False
        self.thread = None
        self.fps = 0.0
        self.frame_count = 0
        self.last_fps_time = time.time()
        
    def start(self):
        """Start the video capture thread."""
        print("Starting threaded video capture...")
        self.cap = cv2.VideoCapture(self.rtsp_url, cv2.CAP_FFMPEG)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.cap.set(cv2.CAP_PROP_FPS, 60)
        
        if not self.cap.isOpened():
            raise RuntimeError("Failed to open video stream")
        
        self.running = True
        self.thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.thread.start()
        print("✅ Video capture thread started")
        
    def _capture_loop(self):
        """Main capture loop running in separate thread."""
        while self.running:
            ret, frame = self.cap.read()
            if ret:
                with self.lock:
                    self.frame = frame.copy()
                    self.frame_count += 1
                    
                    # Calculate FPS
                    current_time = time.time()
                    if current_time - self.last_fps_time >= 1.0:
                        self.fps = self.frame_count / (current_time - self.last_fps_time)
                        self.frame_count = 0
                        self.last_fps_time = current_time
            else:
                time.sleep(0.001)  # Small delay to prevent busy waiting
                
    def read(self):
        """Read the latest frame."""
        with self.lock:
            return None if self.frame is None else self.frame.copy()
            
    def get_fps(self):
        """Get current capture FPS."""
        with self.lock:
            return self.fps
            
    def stop(self):
        """Stop the video capture."""
        self.running = False
        if self.thread:
            self.thread.join(timeout=1.0)
        if self.cap:
            self.cap.release()


class VideoRecorder:
    """Real-time video recorder using separate thread - independent of tracking pipeline."""
    
    def __init__(self, output_dir="recordings"):
        self.output_dir = output_dir
        self.recording = False
        self.writer = None
        self.thread = None
        self.frame_queue = queue.Queue(maxsize=60)  # Larger buffer for 60 FPS
        self.fps = 60  # Recording FPS
        self.frame_size = None
        self.codec = cv2.VideoWriter_fourcc(*'mp4v')
        self.last_frame = None
        self.frame_lock = threading.Lock()
        
        # Recording FPS monitoring
        self.recording_fps = 0.0
        self.recording_frame_count = 0
        self.last_recording_fps_time = time.time()
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
    def start_recording(self, frame_size, fps=60):
        """Start recording with given frame size and FPS."""
        if self.recording:
            return False
            
        self.frame_size = frame_size
        self.fps = fps
        
        # Generate filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"person_tracking_{timestamp}.mp4"
        filepath = os.path.join(self.output_dir, filename)
        
        # Initialize video writer
        self.writer = cv2.VideoWriter(filepath, self.codec, fps, frame_size)
        
        if not self.writer.isOpened():
            print(f"❌ Failed to initialize video writer: {filepath}")
            return False
        
        self.recording = True
        self.thread = threading.Thread(target=self._recording_loop, daemon=True)
        self.thread.start()
        
        print(f"✅ Started recording: {filepath}")
        return True
    
    def stop_recording(self):
        """Stop recording and save file."""
        if not self.recording:
            return
            
        self.recording = False
        
        # Wait for thread to finish
        if self.thread:
            self.thread.join(timeout=2.0)
        
        # Release writer
        if self.writer:
            self.writer.release()
            self.writer = None
        
        print("✅ Recording stopped and saved")
    
    def add_frame(self, frame):
        """Add frame to recording buffer (non-blocking)."""
        if self.recording:
            with self.frame_lock:
                self.last_frame = frame.copy()
    
    def _recording_loop(self):
        """Recording loop running in separate thread - maintains steady 60 FPS."""
        frame_interval = 1.0 / self.fps  # Time between frames for 60 FPS
        last_frame_time = time.time()
        
        while self.recording:
            current_time = time.time()
            
            # Check if it's time for the next frame
            if current_time - last_frame_time >= frame_interval:
                try:
                    # Get the latest frame
                    with self.frame_lock:
                        if self.last_frame is not None:
                            frame = self.last_frame.copy()
                        else:
                            continue  # Skip if no frame available
                    
                    # Write frame to video
                    if self.writer and self.writer.isOpened():
                        self.writer.write(frame)
                        
                        # Update recording FPS counter
                        self.recording_frame_count += 1
                        if current_time - self.last_recording_fps_time >= 1.0:
                            self.recording_fps = self.recording_frame_count / (current_time - self.last_recording_fps_time)
                            self.recording_frame_count = 0
                            self.last_recording_fps_time = current_time
                    
                    last_frame_time = current_time
                    
                except Exception as e:
                    print(f"Recording error: {e}")
            else:
                # Sleep for a short time to maintain steady FPS
                time.sleep(0.001)
    
    def is_recording(self):
        """Check if currently recording."""
        return self.recording
    
    def get_recording_fps(self):
        """Get current recording FPS."""
        return self.recording_fps


class PersonTracker:
    """Person tracking using Ultralytics track() with TensorRT optimization."""
    
    def __init__(self, model_name='yolov8n.pt', use_tensorrt=True):
        """Initialize the person tracker with TensorRT optimization and caching."""
        if not YOLO_AVAILABLE:
            raise ImportError("Ultralytics not available. Install with: pip install ultralytics")
        
        self.person_class_id = 0  # Person class ID in COCO dataset
        self.use_tensorrt = use_tensorrt and TORCH_AVAILABLE
        
        # Check for cached TensorRT engine
        if self.use_tensorrt:
            engine_path = model_name.replace('.pt', '.engine')
            if os.path.exists(engine_path):
                print(f"🚀 Loading cached TensorRT engine: {engine_path}")
                try:
                    self.model = YOLO(engine_path)
                    print("✅ Cached TensorRT engine loaded successfully!")
                except Exception as e:
                    print(f"⚠️ Failed to load cached engine, falling back to standard model: {e}")
                    self.use_tensorrt = False
            else:
                print(f"📦 No cached engine found, exporting from: {model_name}")
                try:
                    print("🚀 Optimizing model for TensorRT with half precision...")
                    temp_model = YOLO(model_name)
                    # Export returns the path to the engine file
                    engine_path = temp_model.export(
                        format='engine',
                        half=True,  # Half precision (FP16)
                        device=0,   # GPU device
                        workspace=4,  # 4GB workspace
                        verbose=False
                    )
                    # Load the exported engine
                    self.model = YOLO(engine_path)
                    print(f"✅ TensorRT optimization complete! Engine saved as: {engine_path}")
                except Exception as e:
                    print(f"⚠️ TensorRT optimization failed, falling back to standard model: {e}")
                    self.use_tensorrt = False
        
        # Load standard model if TensorRT not used or failed
        if not self.use_tensorrt:
            print(f"Loading standard YOLO model: {model_name}")
            self.model = YOLO(model_name)
        
        # Check GPU availability
        if TORCH_AVAILABLE and torch.cuda.is_available():
            print(f"🚀 GPU available: {torch.cuda.get_device_name(0)}")
            print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        else:
            print("⚠️ No GPU available, using CPU")
        
        # Tracking parameters
        self.tracking_history = deque(maxlen=5)
        self.last_person_center = None
        self.tracking_lost_frames = 0
        self.max_lost_frames = 15
        
        # Performance optimization
        self.prediction_confidence = 0.3
        self.tracking_fps = 0.0
        self.tracking_frame_count = 0
        self.last_tracking_fps_time = time.time()
        
    def track_person(self, frame):
        """Track person using Ultralytics track() function."""
        # Use track() for efficient tracking with persistence
        results = self.model.track(
            frame, 
            persist=True, 
            conf=self.prediction_confidence,
            verbose=False,
            tracker="bytetrack.yaml"
        )
        
        self.tracking_frame_count += 1
        current_time = time.time()
        if current_time - self.last_tracking_fps_time >= 1.0:
            self.tracking_fps = self.tracking_frame_count / (current_time - self.last_tracking_fps_time)
            self.tracking_frame_count = 0
            self.last_tracking_fps_time = current_time
        
        person_centers = []
        confidences = []
        track_ids = []
        
        for result in results:
            if result.boxes is not None and result.boxes.id is not None:
                boxes = result.boxes
                for i, box in enumerate(boxes):
                    # Check if it's a person (class 0)
                    if int(box.cls[0]) == self.person_class_id:
                        # Get bounding box coordinates
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        
                        # Calculate center
                        center_x = int((x1 + x2) / 2)
                        center_y = int((y1 + y2) / 2)
                        
                        person_centers.append((center_x, center_y))
                        confidences.append(float(box.conf[0]))
                        track_ids.append(int(box.id[0]))
        
        # Return the person with highest confidence
        if person_centers:
            best_idx = np.argmax(confidences)
            return person_centers[best_idx], confidences[best_idx], track_ids[best_idx]
        
        return None, 0.0, None
    
    def get_tracking_target(self, frame):
        """Get the target position for camera tracking."""
        person_center, confidence, track_id = self.track_person(frame)
        
        if person_center is not None:
            self.last_person_center = person_center
            self.tracking_lost_frames = 0
            self.tracking_history.append(person_center)
            return person_center, confidence, track_id
        else:
            self.tracking_lost_frames += 1
            return None, 0.0, None, None
    
    def get_tensorrt_status(self):
        """Get TensorRT optimization status."""
        if self.use_tensorrt:
            return "TensorRT (FP16)"
        else:
            return "Standard"
    
    def clear_tensorrt_cache(self):
        """Clear cached TensorRT engine to force re-export."""
        engine_path = 'yolov8n.engine'
        if os.path.exists(engine_path):
            os.remove(engine_path)
            print(f"🗑️ Cleared TensorRT cache: {engine_path}")
            return True
        return False


class AutoFollowCameraThreaded:
    """Threaded camera controller with automatic person following."""
    
    def __init__(self, config_file: str = "PTZController.conf", camera_id: int = 1):
        """Initialize the threaded auto-follow camera."""
        self.config = Config(config_file)
        self.camera_id = camera_id
        self.current_camera = None
        self.video_capture = None
        self.tracker = None
        self.recorder = None
        self.running = False
        
        # Control parameters - ADJUST THESE FOR CAMERA SPEED
        self.pan_speed = 0.5      # Pan speed (0.0-1.0): 0.1=slow, 0.3=medium, 0.5=fast
        self.tilt_speed = 0.5     # Tilt speed (0.0-1.0): 0.1=slow, 0.3=medium, 0.5=fast
        self.zoom_speed = 0.3     # Zoom speed (0.0-1.0): 0.1=slow, 0.3=medium, 0.5=fast
        
        # Tracking parameters - ADJUST THESE FOR SENSITIVITY
        self.target_center_x = 0.5    # Target center X (0.0-1.0): 0.5 = center of frame
        self.target_center_y = 0.5    # Target center Y (0.0-1.0): 0.5 = center of frame
        self.dead_zone = 0.1         # Dead zone (0.0-0.5): 0.05=sensitive, 0.15=balanced, 0.25=less sensitive
        self.smooth_factor = 0.1      # Smoothing (0.0-1.0): 0.1=smooth, 0.2=balanced, 0.5=responsive
        
        # Performance optimization
        self.last_movement_time = 0
        self.movement_interval = 0.1  # Minimum time between movements (100ms)
        
        # Threading
        self.tracking_queue = queue.Queue(maxsize=2)  # Small queue for tracking frames
        self.tracking_thread = None
        self.tracking_running = False
        
        # Initialize camera, tracker, and recorder
        self.initialize_camera()
        self.initialize_tracker()
        self.initialize_recorder()
        
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
    
    def initialize_recorder(self):
        """Initialize the video recorder."""
        print("Initializing video recorder...")
        try:
            self.recorder = VideoRecorder()
            print("✅ Video recorder ready")
        except Exception as e:
            print(f"❌ Failed to initialize recorder: {e}")
            raise
    
    def get_stream_uri(self) -> str:
        """Get the RTSP stream URI."""
        username = self.current_camera._Camera__userid
        password = self.current_camera._Camera__password
        host = self.current_camera.host
        return f"rtsp://{username}:{password}@{host}:554/Stream/Live/101"
    
    def start_video_capture(self):
        """Start the threaded video capture."""
        print("Starting video capture...")
        
        stream_uri = self.get_stream_uri()
        
        try:
            self.video_capture = ThreadedVideoCapture(stream_uri)
            self.video_capture.start()
            print("✅ Video capture started")
            return True
            
        except Exception as e:
            print(f"❌ Error starting video capture: {e}")
            return False
    
    def start_tracking_thread(self):
        """Start the tracking thread."""
        print("Starting tracking thread...")
        self.tracking_running = True
        self.tracking_thread = threading.Thread(target=self._tracking_loop, daemon=True)
        self.tracking_thread.start()
        print("✅ Tracking thread started")
    
    def _tracking_loop(self):
        """Tracking loop running in separate thread."""
        while self.tracking_running:
            try:
                # Get frame from queue (non-blocking)
                frame = self.tracking_queue.get(timeout=0.1)
                
                # Perform tracking
                person_center, confidence, track_id = self.tracker.get_tracking_target(frame)
                
                # Calculate camera movement
                if person_center is not None:
                    height, width = frame.shape[:2]
                    pan_speed, tilt_speed = self.calculate_movement(person_center, width, height)
                    self.move_camera(pan_speed, tilt_speed)
                    
                # Store result for display thread
                self.last_tracking_result = {
                    'person_center': person_center,
                    'confidence': confidence,
                    'track_id': track_id,
                    'tracking_fps': self.tracker.tracking_fps
                }
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Tracking error: {e}")
    
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
        
        # if error_x is positive, set error_x to 1 if negative, set error_x to -1
        if error_x > 0:
            error_x = 1
        else:
            error_x = -1
        
        # if error_y is positive, set error_y to 1 if negative, set error_y to -1
        if error_y > 0:
            error_y = 1
        else:
            error_y = -1
        
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
        """Run the automatic person following with threading."""
        if not self.start_video_capture():
            return
        
        # Initialize tracking result
        self.last_tracking_result = {
            'person_center': None,
            'confidence': 0.0,
            'track_id': None,
            'tracking_fps': 0.0
        }
        
        # Start tracking thread
        self.start_tracking_thread()
        
        print("\n🎯 Threaded Auto-follow mode started!")
        print("The camera will automatically follow detected persons")
        print("Press ESC to exit, SPACE to toggle tracking, V to toggle recording")
        
        tracking_enabled = True
        recording_enabled = False
        frame_count = 0
        last_info_time = time.time()
        
        # Create display window
        cv2.namedWindow('Person Tracking Camera (Threaded)', cv2.WINDOW_AUTOSIZE)
        
        try:
            while self.running:
                # Read frame from capture thread
                frame = self.video_capture.read()
                if frame is None:
                    time.sleep(0.01)
                    continue
                
                frame_count += 1
                height, width = frame.shape[:2]
                
                # Add frame to tracking queue (non-blocking)
                if tracking_enabled:
                    try:
                        self.tracking_queue.put_nowait(frame)
                    except queue.Full:
                        pass  # Skip frame if queue is full
                
                # Add frame to recording queue (non-blocking)
                if recording_enabled:
                    self.recorder.add_frame(frame)
                
                # Get latest tracking result
                tracking_result = self.last_tracking_result
                person_center = tracking_result['person_center']
                confidence = tracking_result['confidence']
                track_id = tracking_result['track_id']
                tracking_fps = tracking_result['tracking_fps']
                
                # Draw tracking info
                if person_center is not None:
                    cv2.circle(frame, person_center, 10, (0, 255, 0), 2)
                    cv2.putText(frame, f"Person: {confidence:.2f}", (10, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    cv2.putText(frame, f"ID: {track_id}", (10, 60), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    cv2.putText(frame, "TRACKING", (10, 90), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                else:
                    if self.tracker.tracking_lost_frames > self.tracker.max_lost_frames:
                        self.move_camera(0, 0)  # Stop camera
                    
                    cv2.putText(frame, "NO PERSON DETECTED", (10, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                    cv2.putText(frame, f"Lost: {self.tracker.tracking_lost_frames}", (10, 60), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                if not tracking_enabled:
                    cv2.putText(frame, "TRACKING DISABLED", (10, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
                
                # Show recording status
                if recording_enabled:
                    cv2.putText(frame, "RECORDING", (width - 150, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                    # Draw recording indicator
                    cv2.circle(frame, (width - 20, 20), 8, (0, 0, 255), -1)
                
                # Draw center crosshair
                center_x = int(width * self.target_center_x)
                center_y = int(height * self.target_center_y)
                cv2.line(frame, (center_x - 20, center_y), (center_x + 20, center_y), (255, 255, 255), 2)
                cv2.line(frame, (center_x, center_y - 20), (center_x, center_y + 20), (255, 255, 255), 2)
                
                # Add FPS and status info
                capture_fps = self.video_capture.get_fps()
                recording_fps = self.recorder.get_recording_fps() if recording_enabled else 0
                cv2.putText(frame, f"Capture FPS: {capture_fps:.1f}", (10, height - 160), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                cv2.putText(frame, f"Tracking FPS: {tracking_fps:.1f}", (10, height - 140), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                if recording_enabled:
                    cv2.putText(frame, f"Recording FPS: {recording_fps:.1f}", (10, height - 120), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
                
                # Show TensorRT status
                tensorrt_status = self.tracker.get_tensorrt_status()
                cv2.putText(frame, f"Model: {tensorrt_status}", (10, height - 100), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                cv2.putText(frame, f"Pan: {self.pan_speed:.1f} Tilt: {self.tilt_speed:.1f} Dead: {self.dead_zone:.2f}", (10, height - 80), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
                cv2.putText(frame, f"Frame: {frame_count}", (10, height - 60), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                cv2.putText(frame, "ESC=Exit, SPACE=Toggle, V=Record, H=Home, S=Stop", (10, height - 40), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                cv2.putText(frame, "1/2=Pan 3/4=Tilt 5/6=Dead 7/8=Zoom 9/0=Size R=Reset C=ClearCache", (10, height - 20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                cv2.putText(frame, f"Camera: {self.current_camera.name}", (10, height - 5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                # Display frame
                cv2.imshow('Person Tracking Camera (Threaded)', frame)
                
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
                # Recording controls
                elif key == ord('v'):  # V - Toggle recording
                    if not recording_enabled:
                        # Start recording
                        if self.recorder.start_recording((width, height), 60):
                            recording_enabled = True
                            print("🔴 Recording started")
                        else:
                            print("❌ Failed to start recording")
                    else:
                        # Stop recording
                        self.recorder.stop_recording()
                        recording_enabled = False
                        print("⏹️ Recording stopped")
                # Speed controls
                elif key == ord('1'):  # 1 - Slower pan
                    self.pan_speed = max(0.1, self.pan_speed - 0.1)
                    print(f"Pan speed: {self.pan_speed:.1f}")
                elif key == ord('2'):  # 2 - Faster pan
                    self.pan_speed = min(1.0, self.pan_speed + 0.1)
                    print(f"Pan speed: {self.pan_speed:.1f}")
                elif key == ord('3'):  # 3 - Slower tilt
                    self.tilt_speed = max(0.1, self.tilt_speed - 0.1)
                    print(f"Tilt speed: {self.tilt_speed:.1f}")
                elif key == ord('4'):  # 4 - Faster tilt
                    self.tilt_speed = min(1.0, self.tilt_speed + 0.1)
                    print(f"Tilt speed: {self.tilt_speed:.1f}")
                # Dead zone controls
                elif key == ord('5'):  # 5 - Smaller dead zone (more sensitive)
                    self.dead_zone = max(0.05, self.dead_zone - 0.05)
                    print(f"Dead zone: {self.dead_zone:.2f} (more sensitive)")
                elif key == ord('6'):  # 6 - Larger dead zone (less sensitive)
                    self.dead_zone = min(0.5, self.dead_zone + 0.05)
                    print(f"Dead zone: {self.dead_zone:.2f} (less sensitive)")
                # Reset to defaults
                elif key == ord('r'):  # R - Reset to defaults
                    self.pan_speed = 0.3
                    self.tilt_speed = 0.3
                    self.dead_zone = 0.15
                    self.smooth_factor = 0.2
                    print("Reset to default settings")
                
                # Periodic info
                current_time = time.time()
                if current_time - last_info_time >= 10:  # Every 10 seconds
                    if person_center:
                        rec_fps_info = f" rec_fps:{recording_fps:.1f}" if recording_enabled else ""
                        print(f"Tracking person at ({person_center[0]}, {person_center[1]}) conf:{confidence:.2f} ID:{track_id} cap_fps:{capture_fps:.1f} track_fps:{tracking_fps:.1f}{rec_fps_info}")
                    else:
                        rec_fps_info = f" rec_fps:{recording_fps:.1f}" if recording_enabled else ""
                        print(f"No person detected - cap_fps:{capture_fps:.1f} track_fps:{tracking_fps:.1f}{rec_fps_info}")
                    last_info_time = current_time
                    
        except KeyboardInterrupt:
            print("\nShutting down...")
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Cleanup resources."""
        self.running = False
        self.tracking_running = False
        
        if self.tracking_thread:
            self.tracking_thread.join(timeout=1.0)
        
        if self.video_capture:
            self.video_capture.stop()
        
        if self.recorder and self.recorder.is_recording():
            self.recorder.stop_recording()
        
        cv2.destroyAllWindows()
        
        # Stop camera movement
        if self.current_camera and self.current_camera.isconnected:
            try:
                self.current_camera.stop()
            except:
                pass
    
    def run(self):
        """Main run method."""
        print("THREADED PERSON TRACKING CAMERA CONTROLLER")
        print("==========================================")
        
        if not YOLO_AVAILABLE:
            print("❌ Ultralytics not available. Install with: pip install ultralytics")
            return
        
        self.running = True
        self.run_auto_follow()


def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Threaded Person Tracking Camera Controller')
    parser.add_argument('--camera', type=int, default=1, help='Camera ID to use (default: 1)')
    parser.add_argument('--config', type=str, default='PTZController.conf', help='Configuration file')
    
    args = parser.parse_args()
    
    try:
        controller = AutoFollowCameraThreaded(args.config, args.camera)
        controller.run()
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
