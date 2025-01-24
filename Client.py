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
import threading
import time
from queue import Queue
from dataclasses import dataclass
from typing import List, Optional
from enum import Enum

@dataclass
class MovementCommand:
    command: str
    duration: float
    callback: Optional[callable] = None

class MovementExecutor:
    def __init__(self, robot_client):
        self.robot_client = robot_client
        self.command_queue = Queue()
        self.is_running = False
        self.executor_thread = None
        self.current_sequence = None
        
    def start(self):
        if not self.is_running:
            self.is_running = True
            self.executor_thread = threading.Thread(target=self._process_commands)
            self.executor_thread.daemon = True
            self.executor_thread.start()
            
    def stop(self):
        self.is_running = False
        if self.executor_thread:
            self.executor_thread.join()
            
    def _process_commands(self):
        while self.is_running:
            try:
                if not self.command_queue.empty():
                    command = self.command_queue.get()
                    self.robot_client.send_command(command.command)
                    if command.duration > 0:
                        time.sleep(command.duration)
                    if command.callback:
                        command.callback()
                else:
                    time.sleep(0.01)
            except Exception as e:
                print(f"Error executing movement: {str(e)}")
                
    def queue_movement(self, movement: str, callback: Optional[callable] = None):
        """Queue a single movement command"""
        duration = 0
        if movement.upper() == 'F':
            duration = 2.7
        elif movement.upper() == 'B':
            duration = 2.8
        elif movement.upper() in ['L', 'R']:
            duration = 1.557
            
        self.command_queue.put(MovementCommand(movement.upper(), duration, callback))
        if movement.upper() in ['L', 'R']:
            # Add the small forward movement after turns
            self.command_queue.put(MovementCommand('F', 0.2))
            self.command_queue.put(MovementCommand('S', 0))
            
        # Add stop command after movement
        self.command_queue.put(MovementCommand('S', 0))
        
    def queue_sequence(self, movements: List[str], progress_callback=None):
        """Queue a sequence of movements"""
        total_moves = len(movements)
        for i, move in enumerate(movements):
            def make_callback(index=i):
                if progress_callback:
                    progress_callback((index + 1) / total_moves)
            self.queue_movement(move, callback=make_callback)
            # Add small delay between sequences
            self.command_queue.put(MovementCommand('S', 0.5))

def execute_movement_sequence(robot_client, movements: List[str], progress_callback=None):
    """Execute a sequence of movements asynchronously"""
    executor = MovementExecutor(robot_client)
    executor.start()
    executor.queue_sequence(movements, progress_callback)
    return executor
# Grid and Automation Support Functions
def cartesian_to_matrix_index(coord, grid_dim):
    x, y = coord
    n, m = grid_dim
    return y - (m-1), x

def matrix_index_to_cartesian(coord, grid_dim):
    y, x = coord
    n, m = grid_dim
    return x, (m-1) - y

def automate_inputs(pose: tuple[tuple[int, int], str], measurement_position: tuple[int, int], grid_dim: tuple[int, int]):
    indexes, theta1 = pose
    x1, y1 = matrix_index_to_cartesian(indexes, grid_dim)
    indexes = measurement_position 
    x2, y2 = matrix_index_to_cartesian(indexes, grid_dim)
    theta1 = theta1.lower()
    
    x_disp = x2 - x1
    y_disp = y2 - y1
    movements = []
    curr_theta = theta1

    if theta1 == "u":
        if y_disp > 0:
            movements.extend("f" * y_disp)
        elif y_disp < 0:
            movements.extend("b" * abs(y_disp))
        if x_disp > 0:
            movements.extend("r")
            movements.extend("f" * x_disp)
            curr_theta = "r"
        elif x_disp < 0:
            movements.extend("l")
            movements.extend("f" * abs(x_disp))
            curr_theta = "l"

    elif theta1 == "d":
        if y_disp < 0:
            movements.extend("f" * abs(y_disp))
        elif y_disp > 0:
            movements.extend("b" * y_disp)
        if x_disp < 0:
            movements.extend("r")
            movements.extend("f" * abs(x_disp))
            curr_theta = "l"
        elif x_disp > 0:
            movements.extend("l")
            movements.extend("f" * x_disp)
            curr_theta = "r"

    elif theta1 == "r":
        if x_disp > 0:
            movements.extend("f" * x_disp)
        elif x_disp < 0:
            movements.extend("b" * abs(x_disp))
        if y_disp < 0:
            movements.extend("r")
            movements.extend("f" * abs(y_disp))
            curr_theta = "d"
        elif y_disp > 0:
            movements.extend("l")
            movements.extend("f" * y_disp)
            curr_theta = "u"

    elif theta1 == "l":
        if x_disp < 0:
            movements.extend("f" * abs(x_disp))
        elif x_disp > 0:
            movements.extend("b" * x_disp)
        if y_disp > 0:
            movements.extend("r")
            movements.extend("f" * y_disp)
            curr_theta = "u"
        elif y_disp < 0:
            movements.extend("l")
            movements.extend("f" * abs(y_disp))
            curr_theta = "d"

    return movements, curr_theta

class GridState(Enum):
    TRANSPARENT = 1
    REFLECTANCE = 2
    STRESSED = 3
    END = 4

class GridOverlayManager:
    def __init__(self):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        self.grid_paths = {
            'transparent': "camera-stuff/transparent_grid.png",
            'reflectance': "camera-stuff/reflectance_grid.png",
            'stressed': "camera-stuff/reflectance_grid_stressed.png"
        }
        
        self.state = GridState.TRANSPARENT
        self.transparency_factor = 0.8
        self.grids = {}
        self.has_alpha = False
        self._load_grids()
    
    def _load_grids(self):
        missing_files = []
        for name, path in self.grid_paths.items():
            if not os.path.exists(path):
                missing_files.append(path)
                continue
                
            grid = cv2.imread(path, cv2.IMREAD_UNCHANGED)
            if grid is None:
                missing_files.append(path)
                continue
                
            self.grids[name] = grid
        
        if missing_files:
            print(f"Missing grid files: {', '.join(missing_files)}")
            return False
            
        self.has_alpha = all(grid.shape[2] == 4 for grid in self.grids.values())
        return True

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
            
            if self.has_alpha and overlay_resized.shape[2] == 4:
                overlay_rgb = overlay_resized[:, :, :3]
                overlay_alpha = (overlay_resized[:, :, 3] / 255.0) * self.transparency_factor
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

class RobotClient:
    def __init__(self, host=None, port=8001):
        self.session = requests.Session()
        self.connected = False
        
        # Try to get URL from Streamlit secrets first
        try:
            self.base_url = st.secrets["ROBOT_URL"]
        except:
            # Fall back to direct connection if no secret is available
            if host:
                self.base_url = f"http://{host}:{port}"
            else:
                self.base_url = None
                
        if not self.base_url:
            st.warning("No robot URL configured. Please set up the tunnel URL in Streamlit secrets.")
        
    def connect(self):
        if not self.base_url:
            st.error("No connection URL available")
            return False
            
        try:
            response = self.session.get(f"{self.base_url}/", timeout=5)
            data = response.json()
            self.connected = data.get('connected', False)
            if self.connected:
                st.success("Connected to robot")
            else:
                st.error("Robot controller not responding")
            return self.connected
        except Exception as e:
            st.error(f"Robot connection failed: {str(e)}")
            self.connected = False
            return False
            
    def disconnect(self):
        self.connected = False
        self.session.close()
        
    def send_command(self, command):
        if not self.connected:
            st.error("Not connected to robot")
            return False
            
        try:
            response = self.session.post(
                f"{self.base_url}/command",
                json={'command': command},
                timeout=5
            )
            response.raise_for_status()
            result = response.json()
            return result['status'] == 'success'
        except Exception as e:
            st.error(f"Command failed: {str(e)}")
            return False

class VideoStreamClient:
    def __init__(self):
        self.client_socket = None
        self.connected = False
        self.frame_queue = queue.Queue(maxsize=30)
        self.stop_event = threading.Event()
        self.fps = 0
        self.frame_count = 0
        self.start_time = time.time()
        
        try:
            self.grid_manager = GridOverlayManager()
        except Exception as e:
            print(f"Error initializing grid overlay: {e}")
            self.grid_manager = None

    def connect(self, host, port):
        try:
            if self.client_socket:
                self.client_socket.close()
                
            self.client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.client_socket.settimeout(5)
            self.client_socket.connect((host, port))
            
            self.client_socket.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
            self.client_socket.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 1024 * 1024)
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
            try:
                self.client_socket.shutdown(socket.SHUT_RDWR)
            except:
                pass
            self.client_socket.close()
        while not self.frame_queue.empty():
            try:
                self.frame_queue.get_nowait()
            except queue.Empty:
                break

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
                self.frame_queue.put_nowait(frame)
            else:
                time.sleep(0.01)
def calculate_movements(start_pos: tuple[int, int], start_orientation: str, end_pos: tuple[int, int], grid_dim: tuple[int, int]):
    """Wrapper for automation function"""
    movements, final_theta = automate_inputs((start_pos, start_orientation), end_pos, grid_dim)
    return movements, final_theta

def front_grid(robot_client):
    """Move one grid forward with correct timing"""
    robot_client.send_command('F')
    time.sleep(2.7)
    robot_client.send_command('S')

def back_grid(robot_client):
    """Move one grid backward with correct timing"""
    robot_client.send_command('B')
    time.sleep(2.8)
    robot_client.send_command('S')

def right_grid(robot_client):
    """Turn right with correct sequence and timing"""
    robot_client.send_command('R')
    time.sleep(1.557)
    robot_client.send_command('S')
    time.sleep(0.2)
    robot_client.send_command('F')
    time.sleep(0.2)
    robot_client.send_command('S')

def left_grid(robot_client):
    """Turn left with correct timing"""
    robot_client.send_command('L')
    time.sleep(1.557)
    robot_client.send_command('S')

def add_automation_panel():
    """Creates the automation control panel for the dashboard"""
    st.subheader("Automation Control")
    
    # Grid dimensions
    grid_dim = (3, 3)
    
    # Start position
    st.write("Start Position:")
    col1, col2, col3 = st.columns(3)
    with col1:
        start_x = st.number_input("X", min_value=0, max_value=grid_dim[0]-1, value=0, key="start_x")
    with col2:
        start_y = st.number_input("Y", min_value=0, max_value=grid_dim[1]-1, value=0, key="start_y")
    with col3:
        start_orientation = st.selectbox(
            "Orientation",
            options=['U', 'D', 'L', 'R'],
            index=0,
            key="start_orientation"
        )
    
    # End position
    st.write("Destination:")
    col1, col2 = st.columns(2)
    with col1:
        end_x = st.number_input("X", min_value=0, max_value=grid_dim[0]-1, value=0, key="end_x")
    with col2:
        end_y = st.number_input("Y", min_value=0, max_value=grid_dim[1]-1, value=0, key="end_y")
    
    # Calculate button
    if st.button("Calculate and Execute Path"):
        start_pos = (start_x, start_y)
        end_pos = (end_x, end_y)
        
        try:
            movements, final_theta = automate_inputs(
                (start_pos, start_orientation.lower()),
                end_pos,
                grid_dim
            )
            
            if movements:
                st.info(f"Calculated movements: {', '.join(movements).upper()}")
                st.info(f"Final orientation will be: {final_theta.upper()}")
                
                # Execute movements if robot is connected
                if st.session_state.get('robot_client') and st.session_state['robot_client'].connected:
                    # Create a progress bar and status text
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    def update_progress(progress):
                        progress_bar.progress(progress)
                        status_text.text(f"Executing movement {int(progress * len(movements))}/{len(movements)}")
                    
                    try:
                        # Initialize movement executor if not exists
                        if 'movement_executor' not in st.session_state:
                            st.session_state['movement_executor'] = None
                            
                        # Stop previous execution if exists
                        if st.session_state['movement_executor']:
                            st.session_state['movement_executor'].stop()
                            
                        # Start new execution
                        st.session_state['movement_executor'] = execute_movement_sequence(
                            st.session_state['robot_client'],
                            movements,
                            progress_callback=update_progress
                        )
                        
                    except Exception as e:
                        st.error(f"Failed executing movement sequence: {str(e)}")
                else:
                    st.warning("Robot not connected. Please connect robot first.")
            else:
                st.warning("No movements needed - already at destination")
                
        except Exception as e:
            st.error(f"Error calculating path: {str(e)}")

def main():

    st.set_page_config(page_title="Robot Control Center", layout="wide")
    st.title("Robot Control Center")
    
    # Initialize session state
    if 'video_client' not in st.session_state:
        st.session_state['video_client'] = VideoStreamClient()
    if 'stream_thread' not in st.session_state:
        st.session_state['stream_thread'] = None
    if 'robot_client' not in st.session_state:
        st.session_state['robot_client'] = None
    if 'overlay_enabled' not in st.session_state:
        st.session_state['overlay_enabled'] = False
    if 'current_state' not in st.session_state:
        st.session_state['current_state'] = GridState.TRANSPARENT
    
    # Layout
    col1, col2 = st.columns([2, 1])
    
    # Control Panel (right column)
    with col2:
        st.header("Control Panel")
        
        # Video Connection
        st.subheader("Video Stream")
        video_host = st.text_input("Video Host", value="localhost")
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
        
        # Robot Connection
        st.subheader("Robot Control")
        robot_host = st.text_input("Robot Host", value="10.1.59.194")
        robot_port = st.number_input("Robot Port", value=8001)
        
        if not st.session_state.get('robot_client'):
            if st.button("Connect Robot"):
                st.session_state['robot_client'] = RobotClient(robot_host, robot_port)
                if st.session_state['robot_client'].connect():
                    st.rerun()
        else:
            if st.button("Disconnect Robot"):
                if st.session_state['robot_client']:
                    st.session_state['robot_client'].disconnect()
                st.session_state['robot_client'] = None
                st.rerun()
        
        # Manual Robot Controls
        if st.session_state.get('robot_client') and st.session_state['robot_client'].connected:
            st.subheader("Manual Controls")
            
            c1, c2, c3 = st.columns(3)
            with c2:
                if st.button("↑", use_container_width=True):
                    st.session_state['robot_client'].send_command('F')
                    
            c1, c2, c3 = st.columns(3)
            with c1:
                if st.button("←", use_container_width=True):
                    st.session_state['robot_client'].send_command('L')
            with c2:
                if st.button("■", use_container_width=True):
                    st.session_state['robot_client'].send_command('S')
            with c3:
                if st.button("→", use_container_width=True):
                    st.session_state['robot_client'].send_command('R')
                    
            c1, c2, c3 = st.columns(3)
            with c2:
                if st.button("↓", use_container_width=True):
                    st.session_state['robot_client'].send_command('B')
            
            # Automation Controls
            with st.expander("Automation Control", expanded=False):
                add_automation_panel()
        
        # Grid Overlay Controls
        if st.session_state['video_client'].grid_manager:
            st.subheader("Grid Overlay")
            overlay_enabled = st.checkbox(
                "Enable Grid Overlay",
                value=st.session_state['overlay_enabled']
            )
            
            if overlay_enabled:
                st.write("Grid Type:")
                col1, col2, col3 = st.columns(3)
                with col1:
                    if st.button("Transparent"):
                        st.session_state['current_state'] = GridState.TRANSPARENT
                        st.session_state['video_client'].grid_manager.set_state(GridState.TRANSPARENT)
                with col2:
                    if st.button("Reflectance"):
                        st.session_state['current_state'] = GridState.REFLECTANCE
                        st.session_state['video_client'].grid_manager.set_state(GridState.REFLECTANCE)
                with col3:
                    if st.button("Stressed"):
                        st.session_state['current_state'] = GridState.STRESSED
                        st.session_state['video_client'].grid_manager.set_state(GridState.STRESSED)
                
                transparency = st.slider(
                    "Grid Transparency",
                    0.0, 1.0, 0.8, 0.1
                )
                st.session_state['video_client'].grid_manager.transparency_factor = transparency
            
            if overlay_enabled != st.session_state['overlay_enabled']:
                st.session_state['overlay_enabled'] = overlay_enabled
                st.rerun()
    
    # Video Display (left column)
    with col1:
        if st.session_state['video_client'].connected:
            video_container = st.empty()
            
            while st.session_state['video_client'].connected:
                try:
                    frame = st.session_state['video_client'].frame_queue.get(timeout=0.5)
                    if frame is not None:
                        # Apply grid overlay if enabled
                        if st.session_state['overlay_enabled'] and st.session_state['video_client'].grid_manager:
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

if __name__ == "__main__":
    main()