import streamlit as st
import cv2
import socket
import json
import time
import threading
import queue
import numpy as np
import requests
import PIL.Image as Image
import os
import struct
import pickle
from enum import Enum
from dataclasses import dataclass
from typing import List, Optional

class GridState(Enum):
    TRANSPARENT = 1
    REFLECTANCE = 2
    STRESSED = 3
    END = 4

class GridOverlayManager:
    def __init__(self):
        self.grid_paths = {
            'transparent': "camera-stuff/transparent_grid.png",
            'reflectance': "camera-stuff/reflectance_grid.png",
            'stressed': "camera-stuff/reflectance_grid_stressed.png"
        }
        self.state = GridState.TRANSPARENT
        self.transparency_factor = 0.8
        self.grids = {}
        self._load_grids()
    
    def _load_grids(self):
        for name, path in self.grid_paths.items():
            if os.path.exists(path):
                grid = cv2.imread(path, cv2.IMREAD_UNCHANGED)
                if grid is not None:
                    self.grids[name] = grid
    
    def overlay_grid(self, frame):
        if not self.grids:
            return frame

        result = frame.copy()
        height, width = frame.shape[:2]
        
        current_grid = None
        if self.state == GridState.TRANSPARENT:
            current_grid = self.grids.get('transparent')
        elif self.state == GridState.REFLECTANCE:
            current_grid = self.grids.get('reflectance')
        elif self.state == GridState.STRESSED:
            current_grid = self.grids.get('stressed')
            
        if current_grid is None:
            return result
            
        try:
            grid_size = min(height, width)
            x_offset = (width - grid_size) // 2
            y_offset = (height - grid_size) // 2
            
            overlay_resized = cv2.resize(current_grid, (grid_size, grid_size))
            y1, y2 = y_offset, y_offset + grid_size
            x1, x2 = x_offset, x_offset + grid_size
            
            if overlay_resized.shape[2] == 4:
                overlay_rgb = overlay_resized[:, :, :3]
                overlay_alpha = overlay_resized[:, :, 3] / 255.0 * self.transparency_factor
                alpha_3d = np.stack([overlay_alpha] * 3, axis=2)
                result[y1:y2, x1:x2] = (
                    overlay_rgb * alpha_3d + 
                    result[y1:y2, x1:x2] * (1 - alpha_3d)
                ).astype(np.uint8)
            else:
                result[y1:y2, x1:x2] = cv2.addWeighted(
                    overlay_resized, self.transparency_factor,
                    result[y1:y2, x1:x2], 1 - self.transparency_factor, 
                    0
                )
            
            return result
            
        except Exception as e:
            print(f"Error applying overlay: {str(e)}")
            return frame

    def set_state(self, new_state):
        if new_state in GridState:
            self.state = new_state

class VideoStreamClient:
    def __init__(self):
        self.client_socket = None
        self.connected = False
        self.frame_queue = queue.Queue(maxsize=30)
        self.stop_event = threading.Event()
        self.fps = 0
        self.frame_count = 0
        self.start_time = time.time()
        self.grid_manager = GridOverlayManager()

    def connect(self, host, port):
        try:
            if self.client_socket:
                self.client_socket.close()
                
            self.client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.client_socket.connect((host, port))
            self.client_socket.settimeout(1.0)
            
            self.connected = True
            self.frame_count = 0
            self.start_time = time.time()
            self.stop_event.clear()
            return True
        except Exception as e:
            st.error(f"Video connection failed: {str(e)}")
            return False

    def disconnect(self):
        self.stop_event.set()
        self.connected = False
        if self.client_socket:
            self.client_socket.close()
        self.frame_queue.queue.clear()

    def receive_frame(self):
        try:
            data = self.client_socket.recv(struct.calcsize("L"))
            if not data:
                return None
                
            msg_size = struct.unpack("L", data)[0]
            frame_data = bytearray()
            
            while len(frame_data) < msg_size:
                packet = self.client_socket.recv(min(msg_size - len(frame_data), 4096))
                if not packet:
                    return None
                frame_data.extend(packet)
                
            encoded_frame = pickle.loads(bytes(frame_data))
            frame = cv2.imdecode(np.frombuffer(encoded_frame, np.uint8), cv2.IMREAD_COLOR)
            
            self.frame_count += 1
            elapsed_time = time.time() - self.start_time
            if elapsed_time > 1:
                self.fps = self.frame_count / elapsed_time
                self.frame_count = 0
                self.start_time = time.time()
                
            return frame
            
        except Exception as e:
            return None

    def update_frame(self):
        while not self.stop_event.is_set() and self.connected:
            frame = self.receive_frame()
            if frame is not None:
                if self.frame_queue.full():
                    try:
                        self.frame_queue.get_nowait()
                    except queue.Empty:
                        pass
                self.frame_queue.put(frame)
            else:
                time.sleep(0.01)

def main():
    st.set_page_config(page_title="Enhanced Drone Control", layout="wide")
    st.title("Enhanced Drone Control Center")
    
    if 'video_client' not in st.session_state:
        st.session_state['video_client'] = VideoStreamClient()
    if 'stream_thread' not in st.session_state:
        st.session_state['stream_thread'] = None
    
    col1, col2 = st.columns([2, 1])
    
    with col2:
        st.subheader("Control Panel")
        
        # Video Connection
        st.write("Video Stream")
        video_host = st.text_input("Video Host", "localhost")
        video_port = st.number_input("Video Port", value=8000)
        
        if not st.session_state['video_client'].connected:
            if st.button("Connect Video"):
                if st.session_state['video_client'].connect(video_host, video_port):
                    st.session_state['stream_thread'] = threading.Thread(
                        target=st.session_state['video_client'].update_frame
                    )
                    st.session_state['stream_thread'].daemon = True
                    st.session_state['stream_thread'].start()
                    st.rerun()
        else:
            if st.button("Disconnect Video"):
                st.session_state['video_client'].disconnect()
                st.session_state['stream_thread'] = None
                st.rerun()
        
        # Grid Overlay Controls
        st.subheader("Grid Overlay")
        overlay_enabled = st.checkbox("Enable Grid Overlay")
        
        if overlay_enabled:
            st.write("Grid Type:")
            col1, col2, col3 = st.columns(3)
            with col1:
                if st.button("Transparent"):
                    st.session_state['video_client'].grid_manager.set_state(GridState.TRANSPARENT)
            with col2:
                if st.button("Reflectance"):
                    st.session_state['video_client'].grid_manager.set_state(GridState.REFLECTANCE)
            with col3:
                if st.button("Stressed"):
                    st.session_state['video_client'].grid_manager.set_state(GridState.STRESSED)
            
            transparency = st.slider("Grid Transparency", 0.0, 1.0, 0.8, 0.1)
            st.session_state['video_client'].grid_manager.transparency_factor = transparency
    
    # Video Display
    with col1:
        if st.session_state['video_client'].connected:
            video_container = st.empty()
            
            while st.session_state['video_client'].connected:
                try:
                    frame = st.session_state['video_client'].frame_queue.get(timeout=0.5)
                    if frame is not None:
                        if overlay_enabled:
                            frame = st.session_state['video_client'].grid_manager.overlay_grid(frame)
                        
                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        video_container.image(frame_rgb, channels="RGB", use_container_width=True)
                        
                except queue.Empty:
                    continue
                except Exception as e:
                    st.error(f"Error displaying frame: {str(e)}")
                    break
                
                time.sleep(0.01)
        else:
            st.info("Connect to video stream to begin")

        # NDVI Data Display
        if st.session_state['video_client'].connected:
            try:
                response = requests.get("http://localhost:8001/data/stats")
                if response.status_code == 200:
                    stats = response.json()
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Average NDVI", f"{stats['ndvi_mean']:.3f}")
                    with col2:
                        st.metric("Max NDVI", f"{stats['ndvi_max']:.3f}")
                    with col3:
                        st.metric("Min NDVI", f"{stats['ndvi_min']:.3f}")
            except:
                st.warning("Could not fetch NDVI data")

if __name__ == "__main__":
    main()