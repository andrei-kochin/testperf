import os
import onnx
from typing import List


def _make_input_shape_fixed_fallback(onnx_model: "onnx.ModelProto", input_name: str, shape: List[int]) -> None:
  """
  Pure-onnx fallback for fixing an input tensor shape.

  This mirrors the behavior we want from:
    onnxruntime.tools.onnx_model_utils.make_input_shape_fixed
  but avoids requiring `onnxruntime` just to edit the ONNX graph.
  """
  for inp in onnx_model.graph.input:
    if inp.name != input_name:
      continue
    dims = inp.type.tensor_type.shape.dim
    for i, v in enumerate(shape[: len(dims)]):
      dims[i].dim_param = ""
      dims[i].dim_value = int(v)
    return

def try_export_model(file_path, batch_size, half_precision=False):
    if not os.path.exists(file_path):
      try:
        yolo_model_path = 'yolov11l.pt'
        if not os.path.exists(yolo_model_path):
          try:
            import urllib.request
            urllib.request.urlretrieve('https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo11l.pt', yolo_model_path)
          except Exception as e:
            raise Exception(f'Failed to download YOLO model {e}')
        if os.path.exists(yolo_model_path):
          from ultralytics import YOLO
          model = YOLO(yolo_model_path)
          #model.export(format='onnx', imgsz=640, batch=batch_size, half=half_precision)
          onnx_model_path = model.export(format="onnx", dynamic=True, imgsz=[640, 640])
          print(f"Model exported to ONNX: {onnx_model_path}")

          onnx_model = onnx.load(onnx_model_path)

          # Assumption: first input is the image tensor
          # Ensure each dimension of imgsz is divisible by 32
          imgsz = [((dim + 31) // 32) * 32 for dim in [640, 640]]
          input_name = onnx_model.graph.input[0].name
          # Prefer onnxruntime's helper if available, otherwise fall back to pure onnx.
          try:
            from onnxruntime.tools.onnx_model_utils import make_input_shape_fixed  # type: ignore

            make_input_shape_fixed(onnx_model.graph, input_name, [batch_size, 3, 640, 640])
          except Exception:
            _make_input_shape_fixed_fallback(onnx_model, input_name, [batch_size, 3, 640, 640])

          # Save the modified ONNX model
          onnx.save(onnx_model, file_path)

          #os.rename(yolo_model_path[:-2] + 'onnx', file_path)
        else:
          raise Exception(f'YOLO model file {yolo_model_path} not found')
      except Exception as e:
        raise Exception(f'Failed to export model {e}')
    pass
