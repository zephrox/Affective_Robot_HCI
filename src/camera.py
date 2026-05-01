import cv2

class Camera:
    def __init__(self, camera_index=0):
        self.cap = cv2.VideoCapture(camera_index)
        # Lower resolution for faster processing
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
    def get_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            return None
        # Flip for a natural mirror effect
        return cv2.flip(frame, 1)
        
    def release(self):
        self.cap.release()