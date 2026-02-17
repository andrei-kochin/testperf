import os
from class_model import Model
import numpy as np
import onnxruntime as ort
from .common import try_export_model

class Model(Model):
  """YOLOv8n FP16 inference with using OpenVINO Execution Provider"""
  def __init__(self):
    super().__init__()
    self.sess = None
    self.sess_data = {'providers': ['OpenVINOExecutionProvider']}
    self.model_path = 'yolov8n_fp16{batch}b.onnx'
    if not self.sess_data['providers'][0] in ort.get_available_providers():
      raise Exception(f'OpenVINO Execution Provider is not available')
  def prepare_batch(self, batch_size):
    file_path = self.get_file_path(self.model_path.format(batch=batch_size))
    try_export_model(file_path, batch_size, half_precision=True)
  def read(self):
    file_path = self.get_file_path(self.model_path.format(batch=self.batch_size))
    self.sess = ort.InferenceSession(file_path, **self.sess_data)
  def prepare(self):
    self.input_data = {
      'images': np.random.randn(self.batch_size, 3, 640, 640).astype(np.float32),
    }
  def inference(self):
    #return self.sess.run(['output0'], input_feed=self.input_data)
    return self.sess.run([], input_feed=self.input_data)
  def shutdown(self):
    pass
