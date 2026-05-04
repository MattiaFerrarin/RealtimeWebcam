import numpy as np
import cv2


class UI:
    def __init__(self):
        pass

    def draw(self, frame, filter_name = None, face_count = None, fps = None):
        frame = frame.copy()

        if filter_name is not None:
            cv2.putText(frame, f"Filter: {filter_name}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

        if face_count is not None:
            cv2.putText(frame, f"Faces: {face_count}", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

        if fps is not None:
            cv2.putText(frame, f"FPS: {int(fps)}", (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

        return frame