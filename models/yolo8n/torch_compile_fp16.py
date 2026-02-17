import os
import torch
import numpy as np
from class_model import Model
from ultralytics import YOLO

class Model(Model):
  """YOLOv8n FP16 inference with using default Torch.Compile"""
  def __init__(self):
    super().__init__()
    self.model = None
    self.device = 'cuda'
    if not torch.cuda.is_available():
      raise Exception('CUDA is not available')
    self.model_path = './yolov8n.pt'
  def read(self):
    if not os.path.exists(self.model_path):
      raise Exception(f'Model file {self.model_path} not found')
    self.model = YOLO(self.model_path)
    self.model.to(self.device)
    # Convert model to half precision (FP16)
    self.model.model = self.model.model.half()
    # Compile the model for improved performance
    self.model.model = torch.compile(self.model.model, mode='max-autotune-no-cudagraphs')
  def prepare(self):
    # Create random input tensor (B, C, H, W) in half precision
    self.input_data = torch.randn(
        self.batch_size, 3, 640, 640,
        dtype=torch.float32,
        device=self.device
    )
    min_val = self.input_data.min()
    max_val = self.input_data.max()
    self.input_data = (self.input_data - min_val) / (max_val - min_val)
    self.input_data = self.input_data.cuda().half()
  def inference(self):
    with torch.no_grad():
      return self.model.model(self.input_data)
  def shutdown(self):
    if self.model is not None:
      del self.model
      torch.cuda.empty_cache() if torch.cuda.is_available() else None
