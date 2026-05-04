import os
import sys
import numpy as np
import time

import cv2
from PIL import Image

from concurrent.futures import ThreadPoolExecutor, TimeoutError
executor = ThreadPoolExecutor(max_workers=1)

sys.path.insert(0, os.path.join(os.getcwd(), ".."))

import ImageProcessorClass
import HelperFunc

def millis(startTime):
    return (time.time() - startTime)*1000

cap = None
processor = None
def setup_camera_and_processor():
    global cap, processor
    cap = cv2.VideoCapture("/dev/video0", cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # prevents lag

    root_path = os.path.abspath(os.getcwd())
    config_path = os.path.join(root_path, "UniMERNet/configs/demo.yaml")
    processor = ImageProcessorClass.ImageProcessor(config_path)

def get_latex_from_image():
    ret, frame = cap.read()
    if not ret:
        print("Frame failed")
        return "", None, None

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5,5), 0)

    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # remove noise
    kernel = np.ones((3,3), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)

    # connect letters
    kernel = np.ones((5,5), np.uint8)
    thresh = cv2.dilate(thresh, kernel, iterations=2)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    cropped = HelperFunc.find_sorrounding_contour_and_crop(frame, contours)

    resized_image = cv2.resize(cropped, (int(cropped.shape[1]*0.25), int(cropped.shape[0]*0.25)), interpolation=cv2.INTER_LINEAR)

    frame_rgb = cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(frame_rgb)

    return processor.process_image(pil_image), frame_rgb, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    
def release_camera():
    cap.release()

if __name__ == "__main__":
    setup_camera_and_processor()

    while True:
        timeAtStart = time.time()

        ret, frame = cap.read()
        if not ret:
            print("Frame failed")
            break

        print("Reading the camera took: ", round(millis(timeAtStart)), " ms")

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5,5), 0)

        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # remove noise
        kernel = np.ones((3,3), np.uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)

        # connect letters
        kernel = np.ones((5,5), np.uint8)
        thresh = cv2.dilate(thresh, kernel, iterations=2)

        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        cropped = HelperFunc.find_sorrounding_contour_and_crop(frame, contours)

        resized_image = cv2.resize(cropped, (int(cropped.shape[1]*0.25), int(cropped.shape[0]*0.25)), interpolation=cv2.INTER_LINEAR)

        print("Processing the camera took: ", round(millis(timeAtStart)), " ms")

        cv2.imshow("Camera", frame)
        cv2.imshow("threshold", thresh)
        cv2.imshow("Cropped rezised", resized_image)

        print("Showing the camera took: ", round(millis(timeAtStart)), " ms")

        frame_rgb = cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(frame_rgb)

        future = executor.submit(processor.process_image, pil_image)
        try:
            latex_code = future.result(timeout=4.0)  # seconds
            print(latex_code)
            print("Recognizing the text took: ", round(millis(timeAtStart)), " ms")
        except TimeoutError:
            print("Inference timed out")

        latex_code = processor.process_image(pil_image)
        print(latex_code)
        print("Recognizing the text took: ", round(millis(timeAtStart)), " ms")

        if cv2.waitKey(1) == 27:#Press ESC in camera app
            break

    cap.release()