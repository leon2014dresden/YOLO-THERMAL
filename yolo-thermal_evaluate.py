from ultralytics import YOLO
import torch
from ultralytics.nn.modules import SPPF
from ultralytics import YOLO
from ultralytics.nn.tasks import DetectionModel
import os

def print_result(model_type):
    model_file = model_type + '.yaml'
    pretrained = '/weights/yolo-thermal.pt'
    if not os.path.isfile(pretrained):
        return
    model = DetectionModel(model_file)
    trained_model = YOLO(pretrained)  # load an official model

    metrics = trained_model.val(save_json=True)  # no arguments needed, dataset and settings remembered
    print(end=5*' ')
    for num in range(50, 100, 5):
        print("map" + str(num), end=7*' ')
    print(end='\n')
    print(metrics.box.p_IoU)


if __name__ == '__main__':
    model_type = "yolo-thermal"
    print_result(model_type)