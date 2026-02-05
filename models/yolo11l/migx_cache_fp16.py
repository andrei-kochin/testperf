import os
from class_model import Model
import numpy as np
import migraphx

class Model(Model):
  def __init__(self):
    super().__init__()
    self.model = None
    self.model_path = 'yolov11l_fp16{batch}b.onnx'
    self.model_description = 'YOLOv11l inference using direct MIGraphX with cache'
  def prepare_batch(self, batch_size):
    file_path = self.get_file_path(self.model_path.format(batch=batch_size))
    if not os.path.exists(file_path):
      try:
        yolo_model_path = 'yolov11l.pt'
        if os.path.exists(yolo_model_path):
          from ultralytics import YOLO
          model = YOLO(yolo_model_path)
          model.export(format='onnx', imgsz=640, batch=batch_size, half=True)
          os.rename(yolo_model_path[:-2] + 'onnx', file_path)
        else:
          raise Exception(f'YOLO model file {yolo_model_path} not found')
      except Exception as e:
        raise Exception(f'Failed to export model {e}')
    # Compile model to MIGraphX binary cache
    cache_path = file_path[:-4] + 'mxr'
    if not os.path.exists(cache_path):
      try:
        model = migraphx.parse_onnx(file_path)
        model.compile(migraphx.get_target("gpu"))
        model.save(cache_path)
        del model
      except Exception as e:
        raise Exception(f'Failed to compile and save MIGraphX model: {e}')
  def read(self):
    file_path = self.get_file_path(self.model_path.format(batch=self.batch_size))
    cache_path = file_path[:-4] + 'mxr'
    self.model = migraphx.load(cache_path)
  def prepare(self):
    self.input_data = np.random.randn(self.batch_size, 3, 640, 640).astype(np.float32)
  def inference(self):
    return self.model.run({'images': self.input_data})
  def shutdown(self):
    if self.model:
      del self.model
      self.model = None
