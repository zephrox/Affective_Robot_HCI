import cv2
import threading
import time
import numpy as np
from collections import deque
from deepface import DeepFace

# --- GLOBALS FOR THREAD COMMUNICATION ---
latest_frame = None
current_emotion = "neutral"
is_running = True

# --- HCI STABILITY MANAGER ---
# Keeps track of the last N emotions to prevent erratic robot twitching
EMOTION_WINDOW_SIZE = 5
emotion_history = deque(maxlen=EMOTION_WINDOW_SIZE)

def get_stable_emotion(new_emotion):
    """Returns the most common emotion in the recent history."""
    emotion_history.append(new_emotion)
    # Find the most frequent emotion in our sliding window
    return max(set(emotion_history), key=emotion_history.count)

# --- AI PERCEPTION THREAD ---
def emotion_worker():
    global latest_frame, current_emotion, is_running
    
    while is_running:
        if latest_frame is not None:
            try:
                # We copy the frame to avoid race conditions with the main thread
                frame_to_analyze = latest_frame.copy()
                
                # DeepFace analyze. enforce_detection=False prevents crashes if no face is found
                results = DeepFace.analyze(
                    frame_to_analyze, 
                    actions=['emotion'], 
                    enforce_detection=False,
                    silent=True # Suppress console spam
                )
                
                # DeepFace can return a list if multiple faces are found; we take the first
                if isinstance(results, list):
                    results = results[0]
                    
                raw_emotion = results['dominant_emotion']
                current_emotion = get_stable_emotion(raw_emotion)
                
            except Exception as e:
                # If analysis fails (e.g., face totally obscured), default to neutral
                pass
                
        # Sleep slightly to free up CPU for the main video thread
        time.sleep(0.1) 

# --- ROBOT ACTUATION (SIMULATED UI) ---
def draw_robot_face(frame, emotion):
    """Draws a simple, expressive robot face on the OpenCV frame."""
    h, w, _ = frame.shape
    
    # Robot Face Background (Top right corner)
    center_x, center_y = w - 150, 150
    radius = 100
    cv2.circle(frame, (center_x, center_y), radius, (50, 50, 50), -1) # Dark gray head
    cv2.circle(frame, (center_x, center_y), radius, (255, 255, 255), 3) # White border
    
    # Eyes
    eye_offset_x = 35
    eye_offset_y = 20
    cv2.circle(frame, (center_x - eye_offset_x, center_y - eye_offset_y), 15, (255, 255, 0), -1)
    cv2.circle(frame, (center_x + eye_offset_x, center_y - eye_offset_y), 15, (255, 255, 0), -1)
    
    # Expression mapping
    if emotion == "happy":
        # Smiling curve
        cv2.ellipse(frame, (center_x, center_y + 20), (40, 30), 0, 0, 180, (0, 255, 0), 5)
    elif emotion == "sad":
        # Frowning curve
        cv2.ellipse(frame, (center_x, center_y + 50), (40, 30), 0, 180, 360, (255, 0, 0), 5)
    elif emotion == "angry":
        # Angry eyebrows and straight mouth
        cv2.line(frame, (center_x - 60, center_y - 45), (center_x - 20, center_y - 30), (0, 0, 255), 5)
        cv2.line(frame, (center_x + 60, center_y - 45), (center_x + 20, center_y - 30), (0, 0, 255), 5)
        cv2.line(frame, (center_x - 40, center_y + 30), (center_x + 40, center_y + 30), (0, 0, 255), 5)
    elif emotion == "surprise":
        # 'O' mouth
        cv2.circle(frame, (center_x, center_y + 35), 20, (0, 255, 255), 5)
    else: # Neutral or other
        # Straight mouth
        cv2.line(frame, (center_x - 30, center_y + 30), (center_x + 30, center_y + 30), (255, 255, 255), 5)
        
    # Text label for clarity
    cv2.putText(frame, f"Robot State: {emotion.upper()}", (center_x - 90, center_y + 130), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

# --- MAIN LOOP ---
def main():
    global latest_frame, is_running
    
    print("Starting webcam...")
    cap = cv2.VideoCapture(0)
    
    # Start the AI perception thread
    print("Starting AI engine...")
    ai_thread = threading.Thread(target=emotion_worker)
    ai_thread.start()
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Flip frame horizontally for a more natural mirror-like interaction
        frame = cv2.flip(frame, 1)
        
        # Update the global frame for the AI thread to read
        latest_frame = frame
        
        # Draw the robot simulation responding to the current emotion
        draw_robot_face(frame, current_emotion)
        
        # Display the result
        cv2.imshow("HCI Emotion Aware Robot", frame)
        
        # Press 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            is_running = False
            break
            
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    ai_thread.join()
    print("System shut down gracefully.")

if __name__ == "__main__":
    main()