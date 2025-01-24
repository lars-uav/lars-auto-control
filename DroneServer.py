import cv2
import socket
import pickle
import struct
import sys
import time
import threading

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
    print("Initializing camera...")
    cap = cv2.VideoCapture(0)
    
    # Optimize camera settings for streaming
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Minimize buffer delay
    
    if not cap.isOpened():
        print("Failed to open camera!")
        return
        
    print(f"Camera opened successfully!")
    print(f"Resolution: {cap.get(cv2.CAP_PROP_FRAME_WIDTH)}x{cap.get(cv2.CAP_PROP_FRAME_HEIGHT)}")
    
    # Create socket server
    try:
        server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server_socket.bind((host, port))
        server_socket.listen(5)
        print(f"Server listening on {host}:{port}")
    except Exception as e:
        print(f"Socket error: {e}")
        cap.release()
        return
        
    # Accept multiple clients
    try:
        while True:
            client_socket, addr = server_socket.accept()
            client_thread = threading.Thread(
                target=handle_client,
                args=(client_socket, addr, cap)
            )
            client_thread.daemon = True
            client_thread.start()
    except KeyboardInterrupt:
        print("\nShutting down server...")
    finally:
        cap.release()
        server_socket.close()

if __name__ == "__main__":
    start_stream_server()
