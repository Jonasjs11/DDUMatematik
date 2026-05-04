import argparse
import os
import sys
import numpy as np

import torch
from PIL import Image

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