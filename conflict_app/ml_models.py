from ultralytics import YOLO
import torch
import timm 


yolo_model = YOLO('conflict_app/ml_models/yolov8x.pt')


def load_swin_model(model_path):
    model = timm.create_model(
        "swin_tiny_patch4_window7_224",
        pretrained=False,
        num_classes=2
    )

   
    checkpoint = torch.load(model_path, map_location='cpu')

 
    if isinstance(checkpoint, dict) and "model" in checkpoint:
        state_dict = checkpoint["model"]
    else:
        state_dict = checkpoint

    
    first_key = next(iter(state_dict.keys()))
    if first_key.startswith("model."):
        state_dict = {k[len("model."):]: v for k, v in state_dict.items()}


    if "head.weight" in state_dict:
        state_dict["head.fc.weight"] = state_dict.pop("head.weight")
        state_dict["head.fc.bias"] = state_dict.pop("head.bias")


    model.load_state_dict(state_dict)
    model.eval()
    return model

swin_model = load_swin_model('conflict_app/ml_models/swin_transformer_tiny_v3.pth')