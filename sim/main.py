import cv2
import pygame
import sys
import os

# Ensure the project root is in the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.camera import Camera
from src.face_detect import FaceDetector 
from src.inference import CognitiveInference
from src.mapping import RobotMapper
from sim.robot import RobotSim
from src.smoothing import EmotionSmoother

def main():
    print("Initializing System...")
    cam = Camera()
    detector = FaceDetector()
    inference = CognitiveInference()
    mapper = RobotMapper(port='COM5') # <-- DOUBLE CHECK YOUR PORT HERE
    robot = RobotSim()
    smoother = EmotionSmoother(window_size=30)

    while True:
        frame = cam.get_frame()
        if frame is None: break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = detector.detector.process(rgb_frame)
        
        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0].landmark
            
            # 1. ANALYZE (Think)
            state, _, _ = inference.analyze(landmarks)
            
            # 2. SMOOTH (Stabilize)
            smoother.add_prediction(state)
            stable_state = smoother.get_stable_emotion()

            # 3. ACT - SIMULATION (Digital Twin)
            params = mapper.get_physical_params(stable_state)
            robot.draw_arm(params["servos"], params["led"])

            # 4. ACT - HARDWARE (Physical TUI) <-- THE MISSING LINK
            mapper.send_to_arduino(stable_state) 
            
            # Visual feedback on webcam
            cv2.putText(frame, f"STATE: {stable_state.upper()}", (50, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow("Cognitive Tracking Debug", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break

    cam.release()
    cv2.destroyAllWindows()
    robot.quit()

if __name__ == "__main__":
    main()