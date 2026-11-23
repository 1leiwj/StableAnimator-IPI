import cv2
from PIL import Image
import numpy as np
import torch
import warnings
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    try:
        from transformers import AutoImageProcessor, AutoModel
    except ImportError:
        AutoImageProcessor = None
        AutoModel = None

class DINOEvaluator:
    def __init__(self, device='cuda'):
        self.device = device
        self.processor = None
        self.model = None
        if AutoImageProcessor is not None and AutoModel is not None:
            self.processor = AutoImageProcessor.from_pretrained('facebook/dinov2-base')
            self.model = AutoModel.from_pretrained('facebook/dinov2-base').to(self.device).eval()
        
    def get_dino_feature(self, img_pil):
        if self.model is None:
            return None
        inputs = self.processor(images=img_pil, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
        # Use CLS token representation
        last_hidden_states = outputs.last_hidden_state
        cls_token = last_hidden_states[:, 0, :]
        return cls_token
        
    def calculate_dino_similarity(self, source_image_path, generated_video_path):
        if self.model is None:
            raise ImportError("transformers is not installed. pip install transformers")
            
        try:
            source_img = Image.open(source_image_path).convert("RGB")
        except Exception:
            return None
            
        source_feat = self.get_dino_feature(source_img)
        source_feat = torch.nn.functional.normalize(source_feat, p=2, dim=1)
        
        cap = cv2.VideoCapture(generated_video_path)
        if not cap.isOpened():
            return None
            
        similarities = []
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_pil = Image.fromarray(frame_rgb)
            
            frame_feat = self.get_dino_feature(frame_pil)
            frame_feat = torch.nn.functional.normalize(frame_feat, p=2, dim=1)
            
            # cosine similarity
            sim = (source_feat * frame_feat).sum().item()
            similarities.append(sim)
            
        cap.release()
        
        if not similarities:
            return None
            
        return float(np.mean(similarities))
