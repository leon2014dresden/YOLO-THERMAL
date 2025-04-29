from ultralytics import YOLO
import os
from IPython.display import display, Image
from IPython import display
display.clear_output()

if __name__ == '__main__':
   # Load the model.
   sizes = ['n', 's', 'm', 'l', 'x']
   dataset_name = 'IR_dataset'
   versions = ["yolov8-SPPFCSPC1-SPDConv1-TripletAttention1"]
   batch = 16

   # Training.
   current_file_path = os.path.abspath(__file__)
   par_dir = os.path.dirname(os.path.dirname(os.path.dirname(current_file_path)))
   file_path = '/dataset/' + dataset_name + '/data.yaml'
   file_abs_path = par_dir + file_path
   for version in versions:
      version = version[7:]
      for size in sizes:
         model_name = 'yolov8' + size + '-' + version + '.yaml'
         model = YOLO(model_name)
         result_name = 'yolov8' + size + '-' + version
         results = model.train(
            data=file_abs_path,
            imgsz=640,
            epochs=100,
            batch=batch,
            workers=4,
            name=result_name,
            device=[0],
            amp=False,
         )
         
   