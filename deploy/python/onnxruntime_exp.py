# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import argparse
from typing import List, Tuple

import cv2
import numpy as np
import onnxruntime as ort

from yolo_pose_base import YOLOPOSEBASE

class ONNXPose(YOLOPOSEBASE):
    def __init__(self, model_path: str,  confidence_thres: float, iou_thres: float):
        super(ONNXPose, self).__init__(model_path, confidence_thres, iou_thres)

    def load_model(self) -> None:
        self.model = ort.InferenceSession(self.model_path, providers=["CUDAExecutionProvider", "CPUExecutionProvider"])

        self.model_inputs = self.model.get_inputs()
        # Store the shape of the input for later use
        input_shape = self.model_inputs[0].shape
        self.input_width = input_shape[2]
        self.input_height = input_shape[3]

    def inference(self, input_image: np.ndarray) -> np.ndarray:
        """
        Perform inference using an model and return the boxes, scores and keypoints.

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]: The bounding boxes, scores and keypoints.
        """

        # Preprocess the image data
        img_data, pad = self.preprocess(input_image)

        # Run inference using the preprocessed image data
        outputs = self.model.run(None, {self.model_inputs[0].name: img_data})

        # Perform post-processing on the outputs to obtain output image.
        return self.postprocess(outputs, pad)  # output image


if __name__ == "__main__":
    # Create an argument parser to handle command-line arguments
    parser = argparse.ArgumentParser()
    ASSERT = "D://workspace//deploy_model//yolov8-ncnn-inference//deploy//python//"
    parser.add_argument("--model", type=str, default=ASSERT + "yolo11s-pose.onnx", help="Input your ONNX model.")
    parser.add_argument("--img", type=str, default=ASSERT+"bus.jpg", help="Path to input image.")
    parser.add_argument("--conf-thres", type=float, default=0.5, help="Confidence threshold")
    parser.add_argument("--iou-thres", type=float, default=0.5, help="NMS IoU threshold")
    args = parser.parse_args()

    img = cv2.imread(args.img)

    # Create an instance of the YOLOv8 class with the specified arguments
    detection = ONNXPose(args.model, args.conf_thres, args.iou_thres)

    # Perform object detection and obtain the output image
    (ret_boxes, ret_scores , ret_kpts) = detection.inference(img)
    

    output_image = detection.draw_poses(img,ret_boxes, ret_scores , ret_kpts)
    cv2.imwrite(ASSERT+"ret_onnx.jpg",output_image)