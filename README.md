# YOLOv5 Object Detection using Laptop Camera

## Overview
This project demonstrates how to set up and run a **YOLOv5 object detector** using Python, leveraging a laptop's camera for real-time image capture and object detection. The implementation is written in a Jupyter Notebook and was developed using **Google Colab** for ease of execution.

## Features
- Uses **YOLOv5** for state-of-the-art object detection.
- Captures images directly from the laptop's camera.
- Performs real-time object detection on the captured image.
- Draws bounding boxes and labels on detected objects.
- Works efficiently on CPU and GPU environments.

## Installation
To run this project locally, install the following dependencies:

```sh
pip install torch torchvision cython
pip install 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'
```

Clone the **YOLOv5** repository and install additional requirements:

```sh
git clone https://github.com/ultralytics/yolov5
cd yolov5
pip install -r requirements.txt
```

## Usage
### 1. Import Required Libraries
```python
import torch
import torchvision
from torchvision.transforms import functional as F
from PIL import Image
import cv2
from google.colab.patches import cv2_imshow
```

### 2. Load YOLOv5 Model
```python
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True).to(device)
model.eval()
```

### 3. Capture Image from Laptop Camera
```python
from IPython.display import display, Javascript
from google.colab.output import eval_js
from base64 import b64decode

def take_photo(filename='photo.jpg', quality=0.8):
    js = Javascript('''
        async function takePhoto(quality) {
            const div = document.createElement('div');
            const capture = document.createElement('button');
            capture.textContent = 'Capture';
            div.appendChild(capture);
            
            const video = document.createElement('video');
            video.style.display = 'block';
            const stream = await navigator.mediaDevices.getUserMedia({video: true});

            document.body.appendChild(div);
            div.appendChild(video);
            video.srcObject = stream;
            await video.play();
            
            google.colab.output.setIframeHeight(document.documentElement.scrollHeight, true);
            await new Promise((resolve) => capture.onclick = resolve);
            
            const canvas = document.createElement('canvas');
            canvas.width = video.videoWidth;
            canvas.height = video.videoHeight;
            canvas.getContext('2d').drawImage(video, 0, 0);
            stream.getVideoTracks()[0].stop();
            div.remove();
            return canvas.toDataURL('image/jpeg', quality);
        }
    ''')
    display(js)
    data = eval_js('takePhoto({})'.format(quality))
    binary = b64decode(data.split(',')[1])
    with open(filename, 'wb') as f:
        f.write(binary)
    return filename

image_path = take_photo()
```

### 4. Run Object Detection
```python
def preprocess_image(image_path):
    image = Image.open(image_path)
    image_tensor = F.to_tensor(image)
    return image_tensor.unsqueeze(0).to(device)

image_tensor = preprocess_image(image_path)
outputs = model(image_tensor)
```

### 5. Visualize Detection Results
```python
from yolov5.utils.general import non_max_suppression

def draw_boxes(image_path, outputs, threshold=0.3):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    h, w, _ = image.shape

    for box in outputs:
        score, label, x1, y1, x2, y2 = box[4].item(), int(box[5].item()), box[0].item(), box[1].item(), box[2].item(), box[3].item()
        if score > threshold:
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)
            text = f"{model.names[label]}: {score:.2f}"
            cv2.putText(image, text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    return image

outputs = non_max_suppression(outputs)[0]
result_image = draw_boxes(image_path, outputs)
cv2_imshow(result_image)
```

## Results
The script captures an image from the laptop camera, processes it through YOLOv5, and overlays bounding boxes with labels on detected objects. The processed image is then displayed with highlighted objects.

### sample Result:![1](https://github.com/user-attachments/assets/74401a62-8c0a-415f-a006-af429d85b7f2)

## Contributing
Contributions are welcome! Feel free to submit issues or pull requests to improve the project.

## License
This project is open-source and available under the **MIT License**.

---

### Acknowledgments
This project was implemented using **YOLOv5** by taking learnings from [Ultralytics](https://github.com/ultralytics/yolov5).

