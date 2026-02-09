import os

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
          model.export(format='onnx', imgsz=640, batch=batch_size, half=half_precision)
          os.rename(yolo_model_path[:-2] + 'onnx', file_path)
        else:
          raise Exception(f'YOLO model file {yolo_model_path} not found')
      except Exception as e:
        raise Exception(f'Failed to export model {e}')
    pass
