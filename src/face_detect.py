import cv2
import mediapipe as mp

class FaceDetector:
    def __init__(self, min_detection_confidence=0.5, min_tracking_confidence=0.5):
        # Upgrade to FaceMesh for extreme rotation tolerance
        self.mp_face_mesh = mp.solutions.face_mesh
        self.detector = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=False,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )
        
    def detect(self, frame):
        """Returns a stable bounding box even when the face is turned to the side."""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.detector.process(rgb_frame)
        
        if not results.multi_face_landmarks:
            return None
            
        h, w, _ = frame.shape
        landmarks = results.multi_face_landmarks[0]
        
        # 1. Extract raw X and Y coordinates for all 468 points
        x_coords = [int(landmark.x * w) for landmark in landmarks.landmark]
        y_coords = [int(landmark.y * h) for landmark in landmarks.landmark]
        
        # 2. Find the extreme edges of the face to create a tight box
        x_min, x_max = min(x_coords), max(x_coords)
        y_min, y_max = min(y_coords), max(y_coords)
        
        # 3. Add artificial padding so the Emotion AI can see the whole head/jaw
        padding_x = int((x_max - x_min) * 0.15)
        padding_y = int((y_max - y_min) * 0.15)
        
        # 4. Calculate final safe coordinates (preventing out-of-bounds errors)
        x = max(0, x_min - padding_x)
        y = max(0, y_min - int(padding_y * 1.5)) # Extra padding on top for the forehead
        width = min(w - x, (x_max + padding_x) - x)
        height = min(h - y, (y_max + padding_y) - y)
        
        return (x, y, width, height)