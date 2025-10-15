#!/usr/bin/env python3
"""
Dual Camera Test Script - Step by Step
1. Ping both cameras
2. Test individual streams
3. Display both cameras together
"""

import cv2
import numpy as np
import sys
import os
import threading
import time
import subprocess
from typing import Optional, Tuple

# Add the PTZController module to the path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'PTZController'))

from PTZController.camera import Camera
from PTZController.config import Config
from PTZController import logger


class CameraTester:
    def __init__(self, config_file: str = "PTZController.conf"):
        """Initialize the camera tester."""
        self.config = Config(config_file)
        self.cameras = []
        self.stream_readers = []
        
    def step1_ping_cameras(self):
        """Step 1: Ping both cameras to check connectivity."""
        print("=" * 60)
        print("STEP 1: PING CAMERAS")
        print("=" * 60)
        
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
        
        for i, config in enumerate(camera_configs, 1):
            host = config['host']
            name = config['name']
            
            print(f"\nTesting Camera {i}: {name} ({host})")
            
            # Ping the camera
            try:
                result = subprocess.run(['ping', '-c', '3', host], 
                                      capture_output=True, text=True, timeout=10)
                if result.returncode == 0:
                    print(f"✅ {name} ({host}) - REACHABLE")
                    # Extract ping time
                    lines = result.stdout.split('\n')
                    for line in lines:
                        if 'time=' in line:
                            time_part = line.split('time=')[1].split()[0]
                            print(f"   Ping time: {time_part}")
                            break
                else:
                    print(f"❌ {name} ({host}) - NOT REACHABLE")
            except subprocess.TimeoutExpired:
                print(f"❌ {name} ({host}) - PING TIMEOUT")
            except Exception as e:
                print(f"❌ {name} ({host}) - ERROR: {e}")
        
        return camera_configs
    
    def step2_test_individual_streams(self, camera_configs):
        """Step 2: Test individual camera streams."""
        print("\n" + "=" * 60)
        print("STEP 2: TEST INDIVIDUAL STREAMS")
        print("=" * 60)
        
        for i, config in enumerate(camera_configs, 1):
            name = config['name']
            host = config['host']
            username = config['userid']
            password = config['password']
            
            print(f"\nTesting Camera {i}: {name} ({host})")
            
            # Create RTSP URL
            rtsp_url = f"rtsp://{username}:{password}@{host}:554/Stream/Live/101"
            print(f"RTSP URL: {rtsp_url}")
            
            # Test stream
            try:
                cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                cap.set(cv2.CAP_PROP_FPS, 60)
                
                if not cap.isOpened():
                    print(f"❌ {name} - Failed to open stream")
                    continue
                
                print(f"✅ {name} - Stream opened successfully")
                
                # Try to read a few frames
                frame_count = 0
                start_time = time.time()
                
                for j in range(10):  # Try to read 10 frames
                    ret, frame = cap.read()
                    if ret and frame is not None:
                        frame_count += 1
                        if j == 0:  # First frame info
                            print(f"   Frame size: {frame.shape}")
                            print(f"   Frame type: {frame.dtype}")
                    else:
                        print(f"   Failed to read frame {j+1}")
                        break
                
                elapsed = time.time() - start_time
                fps = frame_count / elapsed if elapsed > 0 else 0
                
                if frame_count > 0:
                    print(f"✅ {name} - Successfully read {frame_count} frames")
                    print(f"   FPS: {fps:.1f}")
                else:
                    print(f"❌ {name} - No frames received")
                
                cap.release()
                
            except Exception as e:
                print(f"❌ {name} - Error: {e}")
    
    def step3_display_dual_cameras(self, camera_configs):
        """Step 3: Display both cameras side by side."""
        print("\n" + "=" * 60)
        print("STEP 3: DUAL CAMERA DISPLAY")
        print("=" * 60)
        
        if len(camera_configs) < 2:
            print("❌ Need at least 2 cameras for dual display")
            return
        
        # Create stream readers for both cameras
        stream_readers = []
        
        for i, config in enumerate(camera_configs[:2], 1):  # Only first 2 cameras
            name = config['name']
            host = config['host']
            username = config['userid']
            password = config['password']
            
            rtsp_url = f"rtsp://{username}:{password}@{host}:554/Stream/Live/101"
            
            print(f"Setting up Camera {i}: {name}")
            
            try:
                cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                cap.set(cv2.CAP_PROP_FPS, 60)
                
                if cap.isOpened():
                    stream_readers.append({
                        'cap': cap,
                        'name': name,
                        'host': host
                    })
                    print(f"✅ {name} - Stream ready")
                else:
                    print(f"❌ {name} - Failed to open stream")
                    
            except Exception as e:
                print(f"❌ {name} - Error: {e}")
        
        if len(stream_readers) < 2:
            print("❌ Need at least 2 working streams for dual display")
            return
        
        print(f"\n✅ Ready to display {len(stream_readers)} cameras")
        print("Press ESC to exit, any other key to continue...")
        
        # Create display window
        cv2.namedWindow('Dual Camera Display', cv2.WINDOW_AUTOSIZE)
        
        frame_count = 0
        last_fps_time = time.time()
        
        try:
            while True:
                frames = []
                
                # Read frames from all cameras
                for i, reader in enumerate(stream_readers):
                    ret, frame = reader['cap'].read()
                    if ret and frame is not None:
                        # Resize frame to fit side by side
                        height, width = frame.shape[:2]
                        if width > 640:  # Resize to fit 2 cameras side by side
                            scale = 640 / width
                            new_width = int(width * scale)
                            new_height = int(height * scale)
                            frame = cv2.resize(frame, (new_width, new_height))
                        
                        # Add camera name overlay
                        cv2.putText(frame, f"Camera {i+1}: {reader['name']}", (10, 30), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                        cv2.putText(frame, f"Host: {reader['host']}", (10, 60), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                        
                        frames.append(frame)
                    else:
                        print(f"Failed to read frame from {reader['name']}")
                
                if len(frames) >= 2:
                    # Combine frames side by side
                    combined_frame = np.hstack(frames)
                    
                    # Add overall info
                    cv2.putText(combined_frame, f"Frame: {frame_count}", (10, combined_frame.shape[0] - 40), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                    cv2.putText(combined_frame, "Press ESC to exit", (10, combined_frame.shape[0] - 20), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                    
                    # Display combined frame
                    cv2.imshow('Dual Camera Display', combined_frame)
                    
                    frame_count += 1
                    
                    # Calculate FPS
                    current_time = time.time()
                    if current_time - last_fps_time >= 5:  # Every 5 seconds
                        fps = frame_count / (current_time - last_fps_time)
                        print(f"Display FPS: {fps:.1f}")
                        frame_count = 0
                        last_fps_time = current_time
                
                # Check for exit
                key = cv2.waitKey(1) & 0xFF
                if key == 27:  # ESC
                    break
                    
        except KeyboardInterrupt:
            print("\nShutting down...")
        finally:
            # Cleanup
            for reader in stream_readers:
                reader['cap'].release()
            cv2.destroyAllWindows()
            print("Dual camera display ended")
    
    def run_all_tests(self):
        """Run all test steps."""
        print("DUAL CAMERA TEST SCRIPT")
        print("=======================")
        
        # Step 1: Ping cameras
        camera_configs = self.step1_ping_cameras()
        
        if not camera_configs:
            print("❌ No camera configurations found!")
            return
        
        # Step 2: Test individual streams
        self.step2_test_individual_streams(camera_configs)
        
        # Step 3: Display dual cameras
        self.step3_display_dual_cameras(camera_configs)
        
        print("\n" + "=" * 60)
        print("ALL TESTS COMPLETED")
        print("=" * 60)


def main():
    """Main function."""
    tester = CameraTester()
    tester.run_all_tests()


if __name__ == "__main__":
    main()
