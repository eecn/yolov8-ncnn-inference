# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import argparse
from typing import List, Tuple
import cv2
import numpy as np
import ncnn

from yolo_pose_base import YOLOPOSEBASE

class NCNNPose(YOLOPOSEBASE):
    def __init__(self, model_path: str,  confidence_thres: float, iou_thres: float):
        super(NCNNPose, self).__init__(model_path, confidence_thres, iou_thres)

    def load_model(self) -> None:
        self.model = ncnn.Net()
        self.model.load_param(self.model_path + ".param")
        self.model.load_model(self.model_path + ".bin")
      
        self.input_width = 640
        self.input_height = 640
        #input_name = self.model.input_names()[0]
        #output_name = self.model.output_names()[0]

    def preprocess(self, input_img: np.ndarray) -> Tuple[np.ndarray, Tuple[int, int]]:
        self.img_width = img.shape[1]
        self.img_height = img.shape[0]

        w = self.img_width
        h = self.img_height
        scale = 1.0
        if w > h:
            scale = float(self.input_width) / w
            w = self.input_width
            h = int(h * scale)
        else:
            scale = float(self.input_height) / h
            h = self.input_height
            w = int(w * scale)

        mat_in = ncnn.Mat.from_pixels_resize(img, ncnn.Mat.PixelType.PIXEL_BGR2RGB, self.img_width, self.img_height, w, h)

        # ultralytics/data/augment.py LetterBox(center=True, stride=32)
        wpad = (self.input_width + 31) // 32 * 32 - w
        hpad = (self.input_height + 31) // 32 * 32 - h
        mat_in_pad = ncnn.copy_make_border(
            mat_in,
            hpad // 2,
            hpad - hpad // 2,
            wpad // 2,
            wpad - wpad // 2,
            ncnn.BorderType.BORDER_CONSTANT,
            114.0,
        )
        self.mean_vals = []
        self.norm_vals = [1 / 255.0, 1 / 255.0, 1 / 255.0]
        mat_in_pad.substract_mean_normalize(self.mean_vals, self.norm_vals)
        #mat_in_pad = np.array(mat_in_pad)
        return mat_in_pad, (hpad // 2, wpad // 2)


    def inference(self, input_image: np.ndarray) -> np.ndarray:
        """
        Perform inference using an model and return the boxes, scores and keypoints.

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]: The bounding boxes, scores and keypoints.
        """
        out = []
        # Preprocess the image data
        img_data, pad = self.preprocess(input_image)
        ex = self.model.create_extractor()

        ex.input(self.model.input_names()[0], ncnn.Mat(img_data).clone())

        _, out0 = ex.extract(self.model.output_names()[0])
        out0 = np.array(out0)
    
        outputs = np.expand_dims(np.array(out0), axis=0)
        return self.postprocess(outputs, pad)  # output image


if __name__ == "__main__":
    # Create an argument parser to handle command-line arguments
    parser = argparse.ArgumentParser()
    ASSERT = "D://workspace//deploy_model//yolov8-ncnn-inference//deploy//python//"
    parser.add_argument("--model", type=str, default=ASSERT+"yolo11s-pose_ncnn_model//model.ncnn", help="Input your NCNN model.")
    parser.add_argument("--img", type=str, default=ASSERT+"bus.jpg", help="Path to input image.")
    parser.add_argument("--conf-thres", type=float, default=0.5, help="Confidence threshold")
    parser.add_argument("--iou-thres", type=float, default=0.5, help="NMS IoU threshold")
    args = parser.parse_args()

    img = cv2.imread(args.img)


    # Create an instance of the YOLOv8 class with the specified arguments
    detection = NCNNPose(args.model, args.conf_thres, args.iou_thres)

    # Perform object detection and obtain the output image
    (ret_boxes, ret_scores , ret_kpts) = detection.inference(img)
    

    output_image = detection.draw_poses(img,ret_boxes, ret_scores , ret_kpts)
    cv2.imwrite(ASSERT+"ret_ncnn.jpg",output_image)