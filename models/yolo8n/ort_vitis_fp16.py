import os
from class_model import Model
import numpy as np
import onnxruntime as ort

class Model(Model):
  def __init__(self):
    super().__init__()
    self.sess = None
    self.sess_data = {'providers': ['VitisAIExecutionProvider']}
    self.model_path = 'yolov8n_fp16{batch}b.onnx'
    self.model_description = 'YOLOv8n FP16 inference with using Vitis AI Execution Provider'
    if not self.sess_data['providers'][0] in ort.get_available_providers():
      raise Exception(f'Vitis AI Execution Provider is not available')
  def prepare_batch(self, batch_size):
    file_path = self.get_file_path(self.model_path.format(batch=batch_size))
    if not os.path.exists(file_path):
      try:
        yolo_model_path = 'yolov8n.pt'
        if os.path.exists(yolo_model_path):
          from ultralytics import YOLO
          model = YOLO(yolo_model_path)
          model.export(format='onnx', imgsz=640, batch=batch_size, half=True)
          os.rename(yolo_model_path[:-2] + 'onnx', file_path)
        else:
          raise Exception(f'YOLO model file {yolo_model_path} not found')
      except Exception as e:
        raise Exception(f'Failed to export model {e}')
    pass
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
