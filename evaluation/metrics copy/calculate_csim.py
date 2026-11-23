import cv2
import numpy as np
import torch
import warnings
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    try:
        from insightface.app import FaceAnalysis
    except ImportError:
        FaceAnalysis = None

class CSIMEvaluator:
    def __init__(self, device='cuda'):
        self.device = device
        self.app = None
        if FaceAnalysis is not None:
            self.app = FaceAnalysis(name='buffalo_l')
            ctx_id = 0 if 'cuda' in str(device) else -1
            self.app.prepare(ctx_id=ctx_id, det_size=(640, 640))
        
    def get_face_embedding(self, img):
        if self.app is None:
            return None
        faces = self.app.get(img)
        if len(faces) == 0:
            return None
        # get the largest face
        largest_face = max(faces, key=lambda f: f.bbox[2]*f.bbox[3] - f.bbox[0]*f.bbox[1])
        return largest_face.embedding

    def calculate_csim(self, source_image_path, generated_video_path):
        if self.app is None:
            raise ImportError("insightface is not installed. pip install insightface onnxruntime-gpu")
            
        source_img = cv2.imread(source_image_path)
        if source_img is None:
            return None
            
        source_embedding = self.get_face_embedding(source_img)
        if source_embedding is None:
            return None # Face detection failed in source image
            
        cap = cv2.VideoCapture(generated_video_path)
        if not cap.isOpened():
            return None
            
        similarities = []
        source_norm = np.linalg.norm(source_embedding)
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            frame_embedding = self.get_face_embedding(frame)
            if frame_embedding is not None:
                # Cosine similarity
                frame_norm = np.linalg.norm(frame_embedding)
                sim = np.dot(source_embedding, frame_embedding) / (source_norm * frame_norm)
                similarities.append(sim)
                
        cap.release()
        
        if not similarities:
            return None
            
        return float(np.mean(similarities))
