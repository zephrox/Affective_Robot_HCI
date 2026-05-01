import numpy as np
from collections import deque

class CognitiveInference:
    def __init__(self, fps=30):
        self.baseline_brow_ratio = None
        self.current_state = "neutral"
        
        # We lock the history to the maximum 2-minute duration (3600 frames)
        self.win_1min = int(1 * 60 * fps)
        self.win_2min = int(2 * 60 * fps)
        self.state_history = deque(maxlen=self.win_2min)
        
        self.gaze_buffer = deque(maxlen=30) 
        self.ENTER_FOCUS = 0.006 
        self.EXIT_FOCUS = 0.012 
        self.STRESS_THRESHOLD = 0.94 

    def get_face_metrics(self, landmarks):
        face_width = np.linalg.norm(np.array([landmarks[234].x, landmarks[234].y]) - 
                                    np.array([landmarks[454].x, landmarks[454].y]))
        brow_dist = np.linalg.norm(np.array([landmarks[107].x, landmarks[107].y]) - 
                                   np.array([landmarks[336].x, landmarks[336].y]))
        return brow_dist / face_width

    def analyze(self, landmarks, calibrating=False):
        brow_ratio = self.get_face_metrics(landmarks)
        if calibrating:
            self.baseline_brow_ratio = brow_ratio
            return "calibrating", 0.0, 0.0

        nose_tip = landmarks[1]
        self.gaze_buffer.append((nose_tip.x, nose_tip.y))
        
        gaze_stability = 0.0
        if len(self.gaze_buffer) == 30:
            coords = np.array(self.gaze_buffer)
            gaze_stability = np.std(coords[:, 0]) + np.std(coords[:, 1])

        # Instantaneous classification
        if self.baseline_brow_ratio and brow_ratio < (self.baseline_brow_ratio * self.STRESS_THRESHOLD):
            instant_state = "stressed"
        elif self.current_state == "focused":
            instant_state = "distracted" if gaze_stability > self.EXIT_FOCUS else "focused"
        else:
            if 0 < gaze_stability < self.ENTER_FOCUS:
                instant_state = "focused"
            elif gaze_stability > 0.020:
                instant_state = "distracted"
            else:
                instant_state = "neutral"

        self.state_history.append(instant_state)
        history_list = list(self.state_history)

        # ASYMMETRICAL LOGIC: 1-min to enter Focus, 2-min to leave
        if self.current_state != "focused":
            # Check only the last 1 minute of data for a "Focus" majority
            recent_window = history_list[-self.win_1min:]
            dominant = max(set(recent_window), key=recent_window.count)
            if dominant == "focused":
                self.current_state = "focused"
        else:
            # Must maintain a "Focus" majority over the full 2 minutes to stay in state
            dominant = max(set(history_list), key=history_list.count)
            if dominant != "focused":
                self.current_state = dominant
            
        return self.current_state, 1.0, 0.0