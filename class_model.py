import os

class Model:
  def __init__(self):
    self.batch_size = 1
    self.total_inference_runs = 100
    self.current_inference_run = 0
    pass
  def prepare_batch(self, batch_size):
    pass
  def reset_inference_run(self):
    self.current_inference_run = 0
  def next_inference_run(self):
    if self.current_inference_run >= self.total_inference_runs:
      return False
    self.current_inference_run += 1
    return True
  def read(self):
    pass
  def read1st(self):
    self.read()
  def readnth(self):
    self.read()
  def warm_up(self):
    self.prepare()
    self.inference1st()
  def prepare(self):
    pass
  def inference(self):
    pass
  def inference1st(self):
    self.inference()
  def inferencenth(self):
    self.inference()
  def shutdown(self):
    pass
  def get_file_path(self, file_name):
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), 'temp', file_name)
  def __str__(self):
    return self.__doc__
