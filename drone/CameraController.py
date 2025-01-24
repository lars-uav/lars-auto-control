import cv2
import socket
import pickle
import struct
import numpy as np
import tensorflow as tf
import threading
import time
from dataclasses import dataclass
from typing import Optional
import requests

@dataclass
class DroneData:
    timestamp: float
    ndvi_stats: dict
    inference_results: Optional[list]
    gps_coords: Optional[tuple]

class VideoStreamServer:
    def __init__(self, host='0.0.0.0', port=8000):
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server_socket.bind((host, port))
        self.server_socket.listen(5)
        self.cap = cv2.VideoCapture(0)
        self.configure_camera()
        self.clients = []
        self.frame_processors = []

    def configure_camera(self):
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    def add_frame_processor(self, processor):
        self.frame_processors.append(processor)

    def send_frame(self, client_socket, frame):
        try:
            _, encoded_frame = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 50])
            data = pickle.dumps(encoded_frame)
            message_size = struct.pack("L", len(data))
            client_socket.sendall(message_size + data)
            return True
        except:
            return False

    def handle_client(self, client_socket, addr):
        print(f'Client connected: {addr}')
        self.clients.append(client_socket)
        frame_count = 0
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break

            frame = cv2.resize(frame, (640, 480))
            
            # Process frame every 30 frames
            if frame_count % 30 == 0:
                for processor in self.frame_processors:
                    processor.process_frame(frame)
            
            if not self.send_frame(client_socket, frame):
                break

            frame_count += 1
            time.sleep(0.001)

        client_socket.close()
        self.clients.remove(client_socket)

    def start(self):
        while True:
            client_socket, addr = self.server_socket.accept()
            thread = threading.Thread(target=self.handle_client, args=(client_socket, addr))
            thread.daemon = True
            thread.start()

    def stop(self):
        for client in self.clients:
            client.close()
        self.server_socket.close()
        self.cap.release()

class NDVIProcessor:
    def __init__(self, data_server_url):
        self.data_server_url = data_server_url

    def calculate_ndvi(self, frame):
        nir = frame[:,:,0].astype(float)
        red = frame[:,:,2].astype(float)
        ndvi = np.where((nir + red) != 0, (nir - red) / (nir + red), 0)
        return ndvi

    def process_frame(self, frame):
        ndvi = self.calculate_ndvi(frame)
        stats = {
            'mean': float(np.mean(ndvi)),
            'max': float(np.max(ndvi)),
            'min': float(np.min(ndvi))
        }
        
        data = DroneData(
            timestamp=time.time(),
            ndvi_stats=stats,
            inference_results=None,
            gps_coords=None
        )
        
        try:
            requests.post(f"{self.data_server_url}/data", json=data.__dict__)
        except:
            print("Failed to send data to server")

class ModelProcessor:
    def __init__(self, model_path, data_server_url):
        self.interpreter = tf.lite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        self.data_server_url = data_server_url

    def process_frame(self, frame):
        preprocessed = cv2.resize(frame, 
            (self.input_details[0]['shape'][1], self.input_details[0]['shape'][2]))
        preprocessed = preprocessed / 255.0
        preprocessed = np.expand_dims(preprocessed, 0).astype(np.float32)
        
        self.interpreter.set_tensor(self.input_details[0]['index'], preprocessed)
        self.interpreter.invoke()
        predictions = self.interpreter.get_tensor(self.output_details[0]['index'])
        

def start_drone_server(model_path, data_server_url="http://localhost:8001"):
    video_server = VideoStreamServer()
    ndvi_processor = NDVIProcessor(data_server_url)
    model_processor = ModelProcessor(model_path, data_server_url)
    
    video_server.add_frame_processor(ndvi_processor)
    video_server.add_frame_processor(model_processor)
    
    try:
        video_server.start()
    except KeyboardInterrupt:
        video_server.stop()

if __name__ == "__main__":
    start_drone_server("/home/lars/lars-benchmark/mobilenetv4_conv_small_model_epoch_44.tflite")
