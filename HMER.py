import os
import cv2
from PIL import Image

import ImageProcessorClass

def setup_processor():
    global processor
    root_path = os.path.abspath(os.getcwd())
    config_path = os.path.join(root_path, "UniMERNet/configs/demo.yaml")
    processor = ImageProcessorClass.ImageProcessor(config_path)

def doHMER(cvImage):
    frame_rgb = cv2.cvtColor(cvImage, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(frame_rgb)
    LaTeX = processor.process_image(pil_image)
    return LaTeX