import os
from class_model import Model
import numpy as np
import openvino as ov
from .common import try_export_model

class Model(Model):
  """YOLOv8n FP16 inference with using OpenVINO"""
  def __init__(self):
    super().__init__()
    self.core = ov.Core()
    self.ov_model = None
    self.compiled_model = None
    self.model_path = 'yolov8n_fp16{batch}b.onnx'
  def prepare_batch(self, batch_size):
    file_path = self.get_file_path(self.model_path.format(batch=batch_size))
    try_export_model(file_path, batch_size, half_precision=True)
  def read(self):
    file_path = self.get_file_path(self.model_path.format(batch=self.batch_size))
    self.ov_model = self.core.read_model(file_path)
    self.compiled_model = self.core.compile_model(self.ov_model, 'CPU')
  def prepare(self):
    self.input_data = {
      'images': np.random.randn(self.batch_size, 3, 640, 640).astype(np.float32),
    }
  def inference(self):
    return self.compiled_model(self.input_data)
  def shutdown(self):
    pass
