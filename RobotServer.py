from flask import Flask, request, jsonify
import serial
import time
import logging
import subprocess
import json

# Add this new endpoint to your Flask server

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

app = Flask(__name__)

class RobotController:
    def __init__(self, serial_port='/dev/ttyUSB0', baud_rate=9600):
        self.arduino = None
        self.serial_port = serial_port
        self.baud_rate = baud_rate
        self.connect()
        
    def connect(self):
        """Establish serial connection with error handling"""
        try:
            # Close existing connection if any
            if self.arduino and self.arduino.is_open:
                self.arduino.close()
                
            # List available serial ports for debugging
            import serial.tools.list_ports
            ports = list(serial.tools.list_ports.comports())
            logging.info(f"Available serial ports: {', '.join(str(p) for p in ports)}")
            
            # Connect to Arduino
            self.arduino = serial.Serial(self.serial_port, self.baud_rate, timeout=2)
            time.sleep(2)  # Wait for Arduino connection
            logging.info(f"Connected to Arduino on {self.serial_port}")
            return True
            
        except Exception as e:
            logging.error(f"Error connecting to Arduino: {e}")
            self.arduino = None
            return False
            
    def send_command(self, command):
        """Send raw command to Arduino"""
        if not self.arduino or not self.arduino.is_open:
            if not self.connect():
                return False
                
        try:
            self.arduino.write(command.encode())
            logging.info(f"Sent command: {command}")
            return True
        except Exception as e:
            logging.error(f"Error sending command: {e}")
            return False
            
    def close(self):
        """Close serial connection"""
        if self.arduino and self.arduino.is_open:
            self.arduino.close()
            logging.info("Serial connection closed")

# Global robot controller instance
robot = RobotController()

@app.route('/')
def home():
    """Simple health check endpoint"""
    connected = robot.arduino and robot.arduino.is_open
    return jsonify({
        'status': 'ok',
        'connected': connected
    })

@app.route('/command', methods=['POST'])
def handle_command():
    """Handle incoming robot commands"""
    try:
        command = request.json.get('command', '').upper()
        if command not in ['F', 'B', 'L', 'R', 'S']:
            return jsonify({
                'status': 'error',
                'message': 'Invalid command'
            }), 400
            
        if robot.send_command(command):
            # Add small delay for movement commands
            if command != 'S':
                time.sleep(0.1)
            return jsonify({
                'status': 'success',
                'message': f'Command {command} executed'
            })
        else:
            return jsonify({
                'status': 'error',
                'message': 'Failed to send command'
            }), 500
            
    except Exception as e:
        logging.error(f"Error handling command: {e}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500
    
@app.route('/automate', methods=['POST'])
def handle_automation():
    """Handle automation requests by calling the automation.py script"""
    try:
        data = request.json
        
        # Extract the automation parameters
        pose = data['pose']
        measurement_position = data['measurement_position']
        grid_dim = data['grid_dim']
        
        # Create command to run automation script
        cmd = [
            'python3',
            'automation.py',  # Make sure this path is correct
            '--pose', json.dumps(pose),
            '--target', json.dumps(measurement_position),
            '--grid', json.dumps(grid_dim)
        ]
        
        # Run the automation script
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True
        )
        
        # Parse the output
        if result.returncode == 0:
            # Assuming automation.py prints the movements and final orientation
            output = result.stdout.strip().split('\n')
            movements = output[0].split(',') if output else []
            final_orientation = output[1] if len(output) > 1 else ''
            
            return jsonify({
                'status': 'success',
                'movements': movements,
                'final_orientation': final_orientation
            })
        else:
            return jsonify({
                'status': 'error',
                'message': f"Automation script failed: {result.stderr}"
            }), 500
            
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

if __name__ == '__main__':
    try:
        logging.info("Starting robot server...")
        app.run(host='0.0.0.0', port=8001)
    finally:
        robot.close()