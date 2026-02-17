import os
from class_model import Model
import numpy as np
import migraphx
from .common import try_export_model

class Model(Model):
  """YOLOv11l inference using direct MIGraphX with cache"""
  def __init__(self):
    super().__init__()
    self.model = None
    self.model_path = 'yolov11l_fp16{batch}b.onnx'
  def prepare_batch(self, batch_size):
    file_path = self.get_file_path(self.model_path.format(batch=batch_size))
    try_export_model(file_path, batch_size, half_precision=True)
    # Compile model to MIGraphX binary cache
    cache_path = file_path[:-4] + 'mxr'
    if not os.path.exists(cache_path):
      try:
        model = migraphx.parse_onnx(file_path)
        model.compile(migraphx.get_target("gpu"))
        migraphx.save(model, cache_path)
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
