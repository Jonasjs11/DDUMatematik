import argparse
import os
import sys
import numpy as np
import time

import cv2
import torch
from PIL import Image

from concurrent.futures import ThreadPoolExecutor, TimeoutError

executor = ThreadPoolExecutor(max_workers=1)

sys.path.insert(0, os.path.join(os.getcwd(), ".."))
from unimernet.common.config import Config
import unimernet.tasks as tasks
from unimernet.processors import load_processor

class ImageProcessor:
    def __init__(self, cfg_path):
        self.cfg_path = cfg_path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model, self.vis_processor = self.load_model_and_processor()

    def load_model_and_processor(self):
        args = argparse.Namespace(cfg_path=self.cfg_path, options=None)
        cfg = Config(args)
        task = tasks.setup_task(cfg)
        model = task.build_model(cfg).to(self.device).half()#Half means half precision
        model.eval()#Added by ChatGPT
        vis_processor = load_processor('formula_image_eval', cfg.config.datasets.formula_rec_eval.vis_processor.eval)

        return model, vis_processor

    def process_single_image(self, image_path):
        try:
            raw_image = Image.open(image_path)
        except IOError:
            print(f"Error: Unable to open image at {image_path}")
            return
        # Convert PIL Image to OpenCV format
        open_cv_image = np.array(raw_image)
        # Convert RGB to BGR
        if len(open_cv_image.shape) == 3:
            # Convert RGB to BGR
            open_cv_image = open_cv_image[:, :, ::-1].copy()
        # Display the image using cv2

        image = self.vis_processor(raw_image).unsqueeze(0).to(self.device).half()#Half means half precision
        with torch.inference_mode():
            output = self.model.generate(
                {"image": image},
                do_sample=False,
                temperature=None,
                top_p=None
            )
        pred = output["pred_str"][0]
        #print(f'Prediction:\n{pred}')

        #cv2.imshow('Original Image', open_cv_image)
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()

        return pred
    
    def process_image(self, inputImage):
        image = self.vis_processor(inputImage).unsqueeze(0).to(self.device).half()#Half means half precision
        with torch.inference_mode():
            output = self.model.generate(
                {"image": image},
                do_sample=False,
                temperature=None,
                top_p=None
            )
        pred = output["pred_str"][0]

        return pred

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
    processor = ImageProcessor(config_path)

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

    cropped = find_sorrounding_contour_and_crop(frame, contours)

    resized_image = cv2.resize(cropped, (int(cropped.shape[1]*0.25), int(cropped.shape[0]*0.25)), interpolation=cv2.INTER_LINEAR)

    frame_rgb = cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(frame_rgb)

    return processor.process_image(pil_image), frame_rgb, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
def find_sorrounding_contour_and_crop(frame, contours):
    xs = []
    ys = []

    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        xs.append(x)
        xs.append(x+w)
        ys.append(y)
        ys.append(y+h)
        cv2.rectangle(frame, (x,y), (x+w,y+h), (255,0,0), 2)
        #cv2.rectangle(thresh, (x,y), (x+w,y+h), (255,0,0), 2)

    if contours:
        x1 = min(xs)
        x2 = max(xs)
        y1 = min(ys)
        y2 = max(ys)

        pad = 20
        x1 = max(x1 - pad, 0)
        y1 = max(y1 - pad, 0)
        x2 = min(x2 + pad, frame.shape[1])
        y2 = min(y2 + pad, frame.shape[0])

        cropped = frame[y1:y2, x1:x2]
        cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)
        #cv2.rectangle(thresh, (x1,y1), (x2,y2), (0,255,0), 2)
    else:
        cropped = frame

    return cropped

def find_grouped_contours_and_crop(
    frame,
    contours,
    horizontal_thresh=40,
    vertical_thresh=80,
    pad=20
):
    boxes = []

    # Get bounding boxes
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        boxes.append([x, y, x + w, y + h])

    merged = True
    while merged:
        merged = False
        new_boxes = []

        while boxes:
            box = boxes.pop(0)
            x1, y1, x2, y2 = box

            merged_box = box

            i = 0
            while i < len(boxes):
                bx1, by1, bx2, by2 = boxes[i]

                # compute gap between boxes
                horizontal_gap = min(abs(bx1 - x2), abs(x1 - bx2))
                vertical_gap = min(abs(by1 - y2), abs(y1 - by2))

                # merge if close enough
                dx, dy = rect_distance(merged_box, boxes[i])

                if dx < horizontal_thresh and dy < vertical_thresh:
                    merged_box = [
                        min(x1, bx1),
                        min(y1, by1),
                        max(x2, bx2),
                        max(y2, by2),
                    ]

                    boxes.pop(i)
                    merged = True
                    x1, y1, x2, y2 = merged_box
                else:
                    i += 1

            new_boxes.append(merged_box)

        boxes = new_boxes

    crops = []

    for (x1, y1, x2, y2) in boxes:
        x1 = max(x1 - pad, 0)
        y1 = max(y1 - pad, 0)
        x2 = min(x2 + pad, frame.shape[1])
        y2 = min(y2 + pad, frame.shape[0])

        crop = frame[y1:y2, x1:x2]
        crops.append(crop)

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    return crops

def filter_contours_by_size(contours, minimum_area=75, minimum_wh=5):
    filtered_contours = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        area = cv2.contourArea(cnt)

        # --- HARD FILTERS ---
        if area < minimum_area:          # remove tiny noise (tune: 30–200)
            continue
        if w < minimum_wh or h < minimum_wh:     # remove tiny boxes
            continue

        filtered_contours.append(cnt)
    return filtered_contours

def rect_distance(boxA, boxB):
    ax1, ay1, ax2, ay2 = boxA
    bx1, by1, bx2, by2 = boxB

    # horizontal gap
    if ax2 < bx1:
        dx = bx1 - ax2
    elif bx2 < ax1:
        dx = ax1 - bx2
    else:
        dx = 0  # overlap

    # vertical gap
    if ay2 < by1:
        dy = by1 - ay2
    elif by2 < ay1:
        dy = ay1 - by2
    else:
        dy = 0  # overlap

    return dx, dy
    
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

        cropped = find_sorrounding_contour_and_crop(frame, contours)

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