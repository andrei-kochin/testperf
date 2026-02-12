from .migx_driver_cache_gist import Model as _BaseModel
from .migx_driver_cache_gist import main


class Model(_BaseModel):
  def __init__(self):
    super().__init__()
    self.export_half = True
    self.quantize_fp16 = True
    self.model_description = "YOLOv11l inference using gist-style MIGraphX export/compile (FP16) + Python migraphx cache"


if __name__ == "__main__":
  raise SystemExit(main(default_fp16=True))

