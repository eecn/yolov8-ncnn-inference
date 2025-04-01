# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import argparse
from typing import List, Tuple

import cv2
import numpy as np
import openvino as ov

from yolo_pose_base import YOLOPOSEBASE

import matplotlib
matplotlib.use('Agg')  # 使用非交互式backend
import matplotlib.pyplot as plt


class OPVINOPose(YOLOPOSEBASE):
    def __init__(self, model_path: str,  confidence_thres: float, iou_thres: float):
        super(OPVINOPose, self).__init__(model_path, confidence_thres, iou_thres)

    def load_model(self) -> None:
        core = ov.Core()
        model = core.read_model(self.model_path)

        n,c,h,w = model.input().shape
        model.reshape({model.input().get_any_name(): ov.PartialShape((n, c, h, w))})
        self.input_shape = model.input().shape
        self.input_width = self.input_shape[2]
        self.input_height = self.input_shape[3]

        ppp = ov.preprocess.PrePostProcessor(model)

        ppp.input().tensor().set_element_type(ov.Type.f32).set_layout(ov.Layout('NHWC')) 

        ppp.input().model().set_layout(ov.Layout('NCHW'))

        ppp.output().tensor().set_element_type(ov.Type.f32)

        model = ppp.build()
        self.model = core.compile_model(model, device_name='CPU')

    def inference(self, input_image: np.ndarray) -> np.ndarray:
        """
        Perform inference using an model and return the boxes, scores and keypoints.

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]: The bounding boxes, scores and keypoints.
        """

        # Preprocess the image data
        img_data, pad = self.preprocess(input_image)
        
        # Run inference using the preprocessed image data
        img_data = img_data.transpose(0, 2, 3, 1)
        results = self.model.infer_new_request({0: img_data})

        predictions = next(iter(results.values()))
        outputs = predictions
        # Perform post-processing on the outputs to obtain output image.
        return self.postprocess(outputs, pad)  # output image


if __name__ == "__main__":
    # Create an argument parser to handle command-line arguments
    parser = argparse.ArgumentParser()
    ASSERT = "D://workspace//deploy_model//yolov8-ncnn-inference//deploy//python//"
    parser.add_argument("--model", type=str, default=ASSERT + "yolo11s-pose_openvino_model//yolo11s-pose.xml", help="Input your OpenVINO model.")
    parser.add_argument("--img", type=str, default=ASSERT+"bus.jpg", help="Path to input image.")
    parser.add_argument("--conf-thres", type=float, default=0.5, help="Confidence threshold")
    parser.add_argument("--iou-thres", type=float, default=0.5, help="NMS IoU threshold")
    args = parser.parse_args()

    img = cv2.imread(args.img)

    # Create an instance of the YOLOv8 class with the specified arguments
    detection = OPVINOPose(args.model, args.conf_thres, args.iou_thres)

    # Perform object detection and obtain the output image
    (ret_boxes, ret_scores , ret_kpts) = detection.inference(img)
    
    output_image = detection.draw_poses(img,ret_boxes, ret_scores , ret_kpts)
    cv2.imwrite(ASSERT+"ret_openvino.jpg",output_image)