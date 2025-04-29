import cv2
import time
from .ml_models import yolo_model, swin_model
import torch
import torchvision.transforms as transforms
from PIL import Image


FRAME_RATE = 10
CONFLICT_THRESHOLD = 0.35
prev_time = 0


transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

def gen_frames():
    global prev_time
    cap = cv2.VideoCapture(0)

    while True:
        success, frame = cap.read()
        if not success:
            break

        current_time = time.time()
        if current_time - prev_time >= 1 / FRAME_RATE:
            prev_time = current_time

          
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = yolo_model.predict(frame_rgb, verbose=False)[0]
            
            person_count = sum(1 for box in results.boxes if int(box.cls[0]) == 0)
            conflict = "No"

          
            if person_count > 1:
                img = Image.fromarray(frame_rgb)
                img_tensor = transform(img).unsqueeze(0).to('cpu')
                
                with torch.no_grad():
                    output = swin_model(img_tensor)
                    probabilities = torch.softmax(output, dim=1)
                    conflict_prob = probabilities[0][1].item()
                    
                    # conflict = "Yes" if conflict_prob >= CONFLICT_THRESHOLD else "No"
                    is_conflict = (conflict_prob < CONFLICT_THRESHOLD)
                    conflict = "Yes" if is_conflict else "No"

           
            text = f"People: {person_count} | Conflict: {conflict}"
            cv2.putText(frame, text, (20, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

           
            ret, buffer = cv2.imencode('.jpg', frame)
            yield (b'--frame\r\n'
                  b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

    cap.release()