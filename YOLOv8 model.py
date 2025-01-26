from ultralytics import YOLO
import torch
import os


if __name__ == '__main__':
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    model = YOLO('yolov8m.pt')

    datapath = os.path.join(os.path.dirname(__file__), 'data-yolov8', 'data.yaml')

    model.train(
        data=datapath,
        epochs=250,
        imgsz=640,
        device=device,
        val=True,
        patience=20,
        plots=True
    )

    model.val(
        data=datapath,
        imgsz=640,
        device=device,
        split='test',
        save_json=True,
        plots=True,
        name="Test Results"
    )

    