from collections import deque

class EmotionSmoother:
    def __init__(self, window_size=5):
        """
        Initializes the temporal smoother.
        window_size: Number of frames to keep in memory. 
        Higher = smoother but slightly more delayed response.
        """
        self.window_size = window_size
        self.emotion_history = deque(maxlen=window_size)
        
    def add_prediction(self, emotion):
        """Adds a new prediction to the rolling window."""
        if emotion is not None:
            self.emotion_history.append(emotion)
            
    def get_stable_emotion(self):
        """Returns the majority vote from the rolling window."""
        if len(self.emotion_history) == 0:
            return "neutral" # Default fallback
            
        # Calculate the mode (majority vote)
        return max(set(self.emotion_history), key=self.emotion_history.count)