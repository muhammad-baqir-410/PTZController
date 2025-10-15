#!/usr/bin/env python3
"""
CV2 Camera Controller - Clean FFmpeg Version
Integrates with PTZController project structure
Uses FFmpeg backend for RTSP streaming with low latency
"""

import cv2
import numpy as np
import sys
import os
import threading
import time
from typing import Optional, Tuple
from collections import deque
from pynput import keyboard

# Add the PTZController module to the path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'PTZController'))

from PTZController.camera import Camera
from PTZController.config import Config
from PTZController import logger


class StreamReader:
    """RTSP stream reader using FFmpeg backend."""
    
    def __init__(self, rtsp_url):
        self.rtsp_url = rtsp_url
        self.cap = None
        self.frame = None
        self.lock = threading.Lock()
        self.running = False
        self.fps = 0.0
        self.thread = None

    def _open_ffmpeg(self):
        """Open stream with FFmpeg backend."""
        try:
            cap = cv2.VideoCapture(self.rtsp_url, cv2.CAP_FFMPEG)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Minimal buffer for low latency
            cap.set(cv2.CAP_PROP_FPS, 60)        # Set to 60 FPS
            if cap.isOpened():
                print("Using FFmpeg backend")
                return cap
        except Exception as e:
            print(f"FFmpeg failed: {e}")
        return None

    def start(self):
        """Start the stream reader."""
        print("Opening RTSP stream...")
        self.cap = self._open_ffmpeg()
        
        if self.cap is None:
            raise RuntimeError("Failed to open RTSP stream")
        
        self.running = True
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()
        print("Stream reader started")

    def _loop(self):
        """Main reading loop."""
        t0 = time.time()
        frames = 0
        
        while self.running:
            ok, f = self.cap.read()
            if ok:
                with self.lock:
                    self.frame = f
                frames += 1
                
                # Calculate FPS
                t = time.time()
                if t - t0 >= 1.0:
                    self.fps = frames / (t - t0)
                    frames = 0
                    t0 = t
            else:
                time.sleep(0.01)

    def read(self):
        """Get latest frame."""
        with self.lock:
            return None if self.frame is None else self.frame.copy()

    def stop(self):
        """Stop the stream reader."""
        self.running = False
        if self.thread:
            self.thread.join(timeout=1.0)
        if self.cap:
            self.cap.release()


class CV2CameraControllerAdapted:
    def __init__(self, config_file: str = "PTZController.conf"):
        """Initialize the adapted camera controller."""
        self.config = Config(config_file)
        self.current_camera = None
        self.stream_reader = None
        self.running = False
        
        # Control parameters
        self.pan_speed = 0.3
        self.tilt_speed = 0.3
        self.zoom_speed = 0.3
        self.focus_speed = 0.3
        
        # Initialize camera
        self.initialize_camera()
        
    def initialize_camera(self):
        """Initialize the first camera from configuration."""
        camera_options = {}
        camera_options['id'] = 1
        for key, value in self.config.items('camera1'):
            if key == 'id':
                continue
            camera_options[key] = value
        if 'name' not in camera_options:
            camera_options['name'] = 'PTZCam1'
            
        self.current_camera = Camera(camera_options)
        
        print(f"Initialized camera: {self.current_camera.name}")
        
        # Wait for camera to connect
        print("Waiting for camera to connect...")
        time.sleep(3)
        
        if self.current_camera.isconnected:
            print(f"Connected to camera: {self.current_camera.name} at {self.current_camera.host}:{self.current_camera.port}")
        else:
            print("Camera not connected!")
            return False
        return True
        
    def get_stream_uri(self) -> Optional[str]:
        """Get the RTSP stream URI."""
        if not self.current_camera or not self.current_camera.isconnected:
            return None
            
        username = self.current_camera._Camera__userid
        password = self.current_camera._Camera__password
        host = self.current_camera.host
        
        # Use the known working URL format
        return f"rtsp://{username}:{password}@{host}:554/Stream/Live/101"
        
    def start_stream(self):
        """Start the video stream."""
        if not self.current_camera or not self.current_camera.isconnected:
            print("No camera connected!")
            return False
            
        stream_uri = self.get_stream_uri()
        if not stream_uri:
            print("Could not get stream URI!")
            return False
            
        try:
            self.stream_reader = StreamReader(stream_uri)
            self.stream_reader.start()
            return True
        except Exception as e:
            print(f"Error starting stream: {e}")
            return False
            
    def handle_key(self, key):
        """Handle keyboard input for camera control."""
        if not self.current_camera or not self.current_camera.isconnected:
            return
            
        try:
            if key == 'w':  # Tilt up
                self.current_camera.move_continuous((0, self.tilt_speed, 0))
            elif key == 's':  # Tilt down
                self.current_camera.move_continuous((0, -self.tilt_speed, 0))
            elif key == 'a':  # Pan left
                self.current_camera.move_continuous((-self.pan_speed, 0, 0))
            elif key == 'd':  # Pan right
                self.current_camera.move_continuous((self.pan_speed, 0, 0))
            elif key == 'q':  # Zoom in
                self.current_camera.move_continuous((0, 0, self.zoom_speed))
            elif key == 'e':  # Zoom out
                self.current_camera.move_continuous((0, 0, -self.zoom_speed))
            elif key == 'r':  # Focus in
                self.current_camera.move_focus_continuous(self.focus_speed)
            elif key == 'f':  # Focus out
                self.current_camera.move_focus_continuous(-self.focus_speed)
            elif key == 'h':  # Home
                self.current_camera.go_home()
            elif key == ' ':  # Stop
                self.current_camera.stop()
                self.current_camera.stop_focus()
            elif key == 'p':  # Show presets
                self.show_presets()
            elif key == '0':  # Go to preset
                self.goto_preset()
            elif key.isdigit() and 1 <= int(key) <= 9:  # Set preset
                self.set_preset(int(key))
            elif key == 'esc':
                self.running = False
                
        except Exception as e:
            print(f"Error executing command: {e}")
            
    def show_presets(self):
        """Show available presets."""
        if not self.current_camera or not self.current_camera.isconnected:
            return
            
        try:
            presets = self.current_camera.get_presets()
            print("\nAvailable presets:")
            for preset in presets:
                print(f"  {preset.token}: {preset.Name}")
        except Exception as e:
            print(f"Error getting presets: {e}")
            
    def goto_preset(self):
        """Go to a specific preset."""
        if not self.current_camera or not self.current_camera.isconnected:
            return
            
        try:
            presets = self.current_camera.get_presets()
            if not presets:
                print("No presets available")
                return
                
            print("Enter preset number to go to:")
            for preset in presets:
                print(f"  {preset.token}: {preset.Name}")
                
            # For simplicity, go to first preset
            if presets:
                self.current_camera.goto_preset(presets[0].token)
                print(f"Going to preset: {presets[0].Name}")
                
        except Exception as e:
            print(f"Error going to preset: {e}")
            
    def set_preset(self, preset_num: int):
        """Set a preset at current position."""
        if not self.current_camera or not self.current_camera.isconnected:
            return
            
        try:
            self.current_camera.set_preset(preset_token=preset_num, preset_name=f"Preset {preset_num}")
            print(f"Set preset {preset_num}")
        except Exception as e:
            print(f"Error setting preset: {e}")
            
    def run(self):
        """Main run loop."""
        if not self.current_camera:
            print("No camera available!")
            return
            
        print("Starting PTZ Camera Controller (Adapted)...")
        print("Controls: WASD=pan/tilt, QE=zoom, RF=focus, H=home, SPACE=stop, P=presets, ESC=quit")
        
        if not self.start_stream():
            print("Failed to start stream!")
            return
            
        # Setup keyboard listener
        def on_press(key):
            try:
                if hasattr(key, 'char') and key.char:
                    c = key.char.lower()
                    if c in ['w', 's', 'a', 'd', 'q', 'e', 'r', 'f', 'h', 'p', '0']:
                        self.handle_key(c)
                    elif c == ' ':
                        self.handle_key(' ')
                    elif c.isdigit() and 1 <= int(c) <= 9:
                        self.handle_key(c)
                elif key == keyboard.Key.space:
                    self.handle_key(' ')
                elif key == keyboard.Key.esc:
                    self.handle_key('esc')
            except Exception as e:
                print(f"Key press error: {e}")

        def on_release(key):
            # Stop movement when key is released
            if hasattr(key, 'char') and key.char:
                c = key.char.lower()
                if c in ['w', 's', 'a', 'd', 'q', 'e', 'r', 'f']:
                    self.current_camera.stop()
                    self.current_camera.stop_focus()
            elif key in [keyboard.Key.space]:
                self.current_camera.stop()
                self.current_camera.stop_focus()

        listener = keyboard.Listener(on_press=on_press, on_release=on_release)
        listener.start()
        
        self.running = True
        last_info = time.time()
        
        try:
            # Create OpenCV window
            cv2.namedWindow('PTZ Camera Controller', cv2.WINDOW_AUTOSIZE)
            
            while self.running:
                frame = self.stream_reader.read()
                if frame is None:
                    time.sleep(0.01)
                    continue
                
                # Add overlay
                cv2.putText(frame, f"Camera: {self.current_camera.name}", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.putText(frame, f"FPS: {self.stream_reader.fps:.1f}", (10, 60), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                cv2.putText(frame, "WASD=pan/tilt, QE=zoom, RF=focus, H=home, SPACE=stop, ESC=quit", 
                           (10, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                
                # Display frame
                cv2.imshow('PTZ Camera Controller', frame)
                
                # Check for ESC key in OpenCV window
                if cv2.waitKey(1) & 0xFF == 27:
                    break
                
                # Periodic info
                if time.time() - last_info > 5:
                    print(f"Stream FPS: {self.stream_reader.fps:.1f}")
                    last_info = time.time()
                    
        except KeyboardInterrupt:
            print("\nShutting down...")
        finally:
            self.cleanup()
            listener.stop()
            
    def cleanup(self):
        """Cleanup resources."""
        self.running = False
        if self.stream_reader:
            self.stream_reader.stop()
        cv2.destroyAllWindows()
        
        # Stop camera movement
        if self.current_camera and self.current_camera.isconnected:
            try:
                self.current_camera.stop()
                self.current_camera.stop_focus()
            except:
                pass


def main():
    """Main function."""
    print("PTZ Camera Controller - Clean FFmpeg Version")
    print("=============================================")
    
    # Check if config file exists
    config_file = "PTZController.conf"
    if not os.path.exists(config_file):
        print(f"Configuration file {config_file} not found!")
        return
        
    try:
        controller = CV2CameraControllerAdapted(config_file)
        controller.run()
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
