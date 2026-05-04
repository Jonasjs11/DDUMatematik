import os

import ImageProcessorClass

def setup_processor():
    root_path = os.path.abspath(os.getcwd())
    config_path = os.path.join(root_path, "UniMERNet/configs/demo.yaml")
    processor = ImageProcessorClass.ImageProcessor(config_path)