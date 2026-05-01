import serial

class RobotMapper:
    def __init__(self, port='COM5', baudrate=9600):
        # 1. Posture & LED Mapping for Simulation
        # [Base, Shoulder, Elbow, Gripper]
        self.postures = {
            "focused":    {"servos": [90, 130, 90, 40], "led": (0, 255, 0)},   # Green
            "stressed":   {"servos": [90, 45, 160, 10], "led": (255, 0, 0)},   # Red
            "distracted": {"servos": [45, 90, 90, 90],  "led": (255, 255, 0)}, # Yellow
            "neutral":    {"servos": [90, 90, 90, 90],  "led": (0, 0, 255)}    # Blue
        }

        # 2. Serial Mapping for Arduino Nano
        self.state_to_char = {
            "neutral": 'N',
            "stressed": 'S',
            "focused": 'F',
            "distracted": 'D'
        }

        # 3. Initialize Serial Connection
        try:
            self.ser = serial.Serial(port, baudrate, timeout=1)
            print(f"Connected to Arduino on {port}")
        except Exception as e:
            self.ser = None
            print(f"Warning: Arduino not found on {port}. Simulation will still run. {e}")

    def get_physical_params(self, state):
        """Returns servo angles and colors for the Pygame simulation."""
        return self.postures.get(state, self.postures["neutral"])

    def send_to_arduino(self, state):
            if self.ser and state in self.state_to_char:
                cmd = self.state_to_char[state]
                # If this doesn't print, the function isn't being called!
                print(f"DEBUG: Python sending '{cmd}' for state '{state}'") 
                self.ser.write(cmd.encode())