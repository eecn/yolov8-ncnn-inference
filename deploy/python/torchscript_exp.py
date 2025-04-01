# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import argparse
import cv2
import numpy as np
import torch

from yolo_pose_base import YOLOPOSEBASE

class NCNNPose(YOLOPOSEBASE):
    def __init__(self, model_path: str,  confidence_thres: float, iou_thres: float):
        super(NCNNPose, self).__init__(model_path, confidence_thres, iou_thres)

    def load_model(self) -> None:
        self.model = torch.jit.load(self.model_path)
        self.model.eval()
        self.model = self.model.to("cpu")
        self.input_width = 640
        self.input_height = 640

    def inference(self, input_image: np.ndarray) -> np.ndarray:
        """
        Perform inference using an model and return the boxes, scores and keypoints.

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]: The bounding boxes, scores and keypoints.
        """
        out = []
        # Preprocess the image data
        img_data, pad = self.preprocess(input_image)
        scripted_output = self.model(torch.from_numpy(img_data)) 
        outputs = scripted_output.cpu().numpy()
        return self.postprocess(outputs, pad)  # output image


if __name__ == "__main__":
    # Create an argument parser to handle command-line arguments
    parser = argparse.ArgumentParser()
    ASSERT = "D://workspace//deploy_model//yolov8-ncnn-inference//deploy//python//"
    parser.add_argument("--model", type=str, default=ASSERT+"yolo11s-pose.torchscript", help="Input your Torchscript model.")
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
    cv2.imwrite(ASSERT+"ret_torchscript.jpg",output_image)