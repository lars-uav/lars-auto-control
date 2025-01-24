import cv2
import socket
import pickle
import struct
import sys
import time
import threading
from flask import Flask, jsonify
import numpy as np
from datetime import datetime
import os
import tensorflow as tf

app = Flask(__name__)
drone_server = None
class DroneServer:
    def __init__(self):
        self.cap = None
        self.server_socket = None
        self.latest_frame = None
        self.frame_lock = threading.Lock()
        
        # Set up image capture directory
        self.capture_dir = "captured_images"
        os.makedirs(self.capture_dir, exist_ok=True)
        
        # Load TFLite model
        self.interpreter = self.load_tflite_model()
        if self.interpreter:
            # Get model details
            self.input_details = self.interpreter.get_input_details()
            self.output_details = self.interpreter.get_output_details()
            
    def load_tflite_model(self):
        try:
            interpreter = tf.interpreter.TFLiteInterpreter(model_path='model.tflite')
            interpreter.allocate_tensors()
            return interpreter
        except Exception as e:
            print(f"Error loading TFLite model: {e}")
            return None

    def preprocess_image(self, img):
        """Preprocess image according to model requirements"""
        # Resize to model input size
        img = cv2.resize(img, (640, 480))
        
        # Convert BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Normalize to [0,1]
        img = img.astype(np.float32) / 255.0
        
        # Add batch dimension
        img = np.expand_dims(img, axis=0)
        
        # Transpose to (batch_size, channels, height, width)
        img = np.transpose(img, (0, 3, 1, 2))
        
        return img

    def classify_image(self, image_path):
        try:
            # Read image
            img = cv2.imread(image_path)
            if img is None:
                raise Exception("Failed to read image")

            # Preprocess image
            processed_img = self.preprocess_image(img)

            # Set input tensor
            self.interpreter.set_tensor(self.input_details[0]['index'], processed_img)

            # Run inference
            start_time = time.time()
            self.interpreter.invoke()
            inference_time = (time.time() - start_time) * 1000  # Convert to milliseconds

            # Get output
            output_data = self.interpreter.get_tensor(self.output_details[0]['index'])

            # Process output
            prediction = np.argmax(output_data)
            confidence = float(output_data[0][prediction])

            # Map prediction to class label (replace with your classes)
            classes = ['class1', 'class2', 'class3']  # Update with your actual classes
            result = {
                'class': classes[prediction],
                'confidence': confidence,
                'inference_time_ms': inference_time
            }
            return result
            
        except Exception as e:
            print(f"Error during classification: {e}")
            return {'error': str(e)}

    def capture_and_classify(self):
        with self.frame_lock:
            if self.latest_frame is None:
                return {'error': 'No frame available'}

            # Generate filename with timestamp
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            image_path = os.path.join(self.capture_dir, f'capture_{timestamp}.jpg')

            # Save image
            cv2.imwrite(image_path, self.latest_frame)

            # Run classification
            result = self.classify_image(image_path)
            result['image_path'] = image_path
            return result

def send_frame(client_socket, frame, compress_params=[cv2.IMWRITE_JPEG_QUALITY, 50]):
    """Compress and send a single frame"""
    # Compress frame to JPEG format
    _, encoded_frame = cv2.imencode('.jpg', frame, compress_params)
    data = pickle.dumps(encoded_frame)
    
    # Send frame size followed by data
    try:
        message_size = struct.pack("L", len(data))
        client_socket.sendall(message_size + data)
        return True
    except (ConnectionResetError, BrokenPipeError, socket.error):
        return False

def handle_client(client_socket, addr, cap):
    """Handle individual client connection"""
    print(f'Handling connection from {addr}')
    frame_count = 0
    start_time = time.time()
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to capture frame!")
                break

            # Resize frame to reduce bandwidth usage
            frame = cv2.resize(frame, (640, 480))
                
            frame_count += 1
            if frame_count % 30 == 0:  # Print FPS every 30 frames
                elapsed_time = time.time() - start_time
                fps = frame_count / elapsed_time
                print(f"Streaming to {addr} at {fps:.2f} FPS")
            
            if not send_frame(client_socket, frame):
                print(f"\nClient {addr} disconnected")
                break
            
            # Small sleep to prevent overwhelming the network
            time.sleep(0.001)
            
    except Exception as e:
        print(f"\nError with client {addr}: {e}")
    finally:
        client_socket.close()
        print(f"Connection closed for {addr}")

def start_stream_server(host='0.0.0.0', port=8000):
    global drone_server
    print("Initializing camera...")
    drone_server = DroneServer()
    drone_server.cap = cv2.VideoCapture(0)
    
    # Optimize camera settings
    drone_server.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    drone_server.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    drone_server.cap.set(cv2.CAP_PROP_FPS, 30)
    drone_server.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    
    if not drone_server.cap.isOpened():
        print("Failed to open camera!")
        return
        
    print(f"Camera opened successfully!")
    
    try:
        # Start Flask server in a thread
        flask_thread = threading.Thread(
            target=lambda: app.run(host='0.0.0.0', port=8002, threaded=True)
        )
        flask_thread.daemon = True
        flask_thread.start()
        print(f"Flask server started on port 8002")
        
        # Start video streaming server
        server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server_socket.bind((host, port))
        server_socket.listen(5)
        print(f"Video streaming server listening on {host}:{port}")
        
        while True:
            client_socket, addr = server_socket.accept()
            client_thread = threading.Thread(
                target=handle_client,
                args=(client_socket, addr)
            )
            client_thread.daemon = True
            client_thread.start()
            
    except KeyboardInterrupt:
        print("\nShutting down server...")
    except Exception as e:
        print(f"Error: {str(e)}")
    finally:
        if drone_server.cap:
            drone_server.cap.release()
        if 'server_socket' in locals():
            server_socket.close()

@app.route('/')
def home():
    return jsonify({'status': 'ok'})

@app.route('/capture', methods=['POST'])
def handle_capture():
    global drone_server
    if drone_server is None:
        return jsonify({'error': 'Server not initialized'}), 500
    
    result = drone_server.capture_and_classify()
    return jsonify(result)

if __name__ == "__main__":
    start_stream_server()