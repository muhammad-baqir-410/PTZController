#!/usr/bin/env python3
"""
Dual Camera Display - Improved Version
Handles both cameras with better resource management
"""

import cv2
import numpy as np
import sys
import os
import threading
import time
from typing import Optional, Tuple

# Add the PTZController module to the path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'PTZController'))

from PTZController.camera import Camera
from PTZController.config import Config
from PTZController import logger


class StreamReader:
    """Individual stream reader for each camera."""
    
    def __init__(self, rtsp_url, camera_name):
        self.rtsp_url = rtsp_url
        self.camera_name = camera_name
        self.cap = None
        self.frame = None
        self.lock = threading.Lock()
        self.running = False
        self.fps = 0.0
        self.thread = None
        self.frame_count = 0

    def start(self):
        """Start the stream reader."""
        print(f"Starting stream for {self.camera_name}...")
        
        try:
            self.cap = cv2.VideoCapture(self.rtsp_url, cv2.CAP_FFMPEG)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            self.cap.set(cv2.CAP_PROP_FPS, 60)
            
            if not self.cap.isOpened():
                print(f"❌ {self.camera_name} - Failed to open stream")
                return False
            
            print(f"✅ {self.camera_name} - Stream opened")
            
            self.running = True
            self.thread = threading.Thread(target=self._loop, daemon=True)
            self.thread.start()
            return True
            
        except Exception as e:
            print(f"❌ {self.camera_name} - Error: {e}")
            return False

    def _loop(self):
        """Main reading loop."""
        t0 = time.time()
        frames = 0
        
        while self.running:
            ret, frame = self.cap.read()
            if ret and frame is not None:
                with self.lock:
                    self.frame = frame.copy()
                frames += 1
                self.frame_count += 1
                
                # Calculate FPS every 30 frames
                if frames % 30 == 0:
                    t = time.time()
                    if t - t0 > 0:
                        self.fps = frames / (t - t0)
                        frames = 0
                        t0 = t
            else:
                time.sleep(0.01)  # Small delay on failure

    def get_frame(self):
        """Get latest frame."""
        with self.lock:
            return None if self.frame is None else self.frame.copy()

    def stop(self):
        """Stop the stream reader."""
        self.running = False
        if self.thread:
            self.thread.join(timeout=2.0)
        if self.cap:
            self.cap.release()


class DualCameraDisplay:
    def __init__(self, config_file: str = "PTZController.conf"):
        """Initialize the dual camera display."""
        self.config = Config(config_file)
        self.stream_readers = []
        
    def initialize_cameras(self):
        """Initialize both cameras."""
        print("Initializing cameras...")
        
        camera_configs = []
        for section in self.config.sections():
            if section in ['General', 'Webserver', 'DEFAULT']:
                continue
                
            camera_options = {}
            camera_options['id'] = len(camera_configs) + 1
            for key, value in self.config.items(section):
                if key == 'id':
                    continue
                camera_options[key] = value
            if 'name' not in camera_options:
                camera_options['name'] = section[:12] if len(section) > 12 else section
                
            camera_configs.append(camera_options)
        
        print(f"Found {len(camera_configs)} camera configurations")
        
        # Create stream readers for first 2 cameras
        for i, config in enumerate(camera_configs[:2], 1):
            name = config['name']
            host = config['host']
            username = config['userid']
            password = config['password']
            
            rtsp_url = f"rtsp://{username}:{password}@{host}:554/Stream/Live/101"
            
            reader = StreamReader(rtsp_url, name)
            if reader.start():
                self.stream_readers.append(reader)
                print(f"✅ Camera {i}: {name} ready")
            else:
                print(f"❌ Camera {i}: {name} failed")
        
        return len(self.stream_readers) >= 2
    
    def run_display(self):
        """Run the dual camera display."""
        if not self.initialize_cameras():
            print("❌ Need at least 2 working cameras for dual display")
            return
        
        print(f"\n✅ Ready to display {len(self.stream_readers)} cameras")
        print("Press ESC to exit")
        
        # Create display window
        cv2.namedWindow('Dual Camera Display', cv2.WINDOW_AUTOSIZE)
        
        frame_count = 0
        last_info_time = time.time()
        
        try:
            while True:
                frames = []
                
                # Read frames from all cameras
                for i, reader in enumerate(self.stream_readers):
                    frame = reader.get_frame()
                    if frame is not None:
                        # Resize frame to fit side by side
                        height, width = frame.shape[:2]
                        if width > 640:  # Resize to fit 2 cameras side by side
                            scale = 640 / width
                            new_width = int(width * scale)
                            new_height = int(height * scale)
                            frame = cv2.resize(frame, (new_width, new_height))
                        
                        # Add camera info overlay
                        cv2.putText(frame, f"Camera {i+1}: {reader.camera_name}", (10, 30), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                        cv2.putText(frame, f"FPS: {reader.fps:.1f}", (10, 60), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                        cv2.putText(frame, f"Frames: {reader.frame_count}", (10, 80), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                        
                        frames.append(frame)
                
                if len(frames) >= 2:
                    # Combine frames side by side
                    combined_frame = np.hstack(frames)
                    
                    # Add overall info
                    cv2.putText(combined_frame, f"Display Frame: {frame_count}", (10, combined_frame.shape[0] - 40), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                    cv2.putText(combined_frame, "Press ESC to exit", (10, combined_frame.shape[0] - 20), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                    
                    # Display combined frame
                    cv2.imshow('Dual Camera Display', combined_frame)
                    
                    frame_count += 1
                    
                    # Periodic info
                    current_time = time.time()
                    if current_time - last_info_time >= 5:  # Every 5 seconds
                        print(f"Display running - Camera 1 FPS: {self.stream_readers[0].fps:.1f}, "
                              f"Camera 2 FPS: {self.stream_readers[1].fps:.1f}")
                        last_info_time = current_time
                else:
                    print("Waiting for frames...")
                    time.sleep(0.1)
                
                # Check for exit
                key = cv2.waitKey(1) & 0xFF
                if key == 27:  # ESC
                    break
                    
        except KeyboardInterrupt:
            print("\nShutting down...")
        finally:
            # Cleanup
            for reader in self.stream_readers:
                reader.stop()
            cv2.destroyAllWindows()
            print("Dual camera display ended")
    
    def run(self):
        """Main run method."""
        print("DUAL CAMERA DISPLAY")
        print("==================")
        self.run_display()


def main():
    """Main function."""
    display = DualCameraDisplay()
    display.run()


if __name__ == "__main__":
    main()
