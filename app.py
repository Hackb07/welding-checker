import gradio as gr
from ultralytics import YOLO
import cv2
from PIL import Image
import numpy as np

# Load YOLO model
model = YOLO("welding.pt")

# Detection function
def detect_welding(image: Image.Image) -> Image.Image:
    # Convert PIL to OpenCV format
    img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    # Run YOLO detection
    results = model(img_cv, verbose=False)
    annotated_frame = results[0].plot()

    # Convert OpenCV BGR to PIL RGB for display
    annotated_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
    return Image.fromarray(annotated_rgb)

# Gradio Interface
interface = gr.Interface(
    fn=detect_welding,
    inputs=gr.Image(type="pil", label="Upload Welding Image"),
    outputs=gr.Image(type="pil", label="Detected Image"),
    title="🖼️ Welding Photo Detection",
    description="Upload a welding image and detect using YOLOv8 custom model.",
    theme="default"
)

if __name__ == "__main__":
    interface.launch(share=True)
