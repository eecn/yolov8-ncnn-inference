# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import argparse
from typing import List, Tuple

import cv2
import numpy as np

from ultralytics.utils import ASSETS, yaml_load
from ultralytics.utils.checks import  check_yaml
from abc import ABC, abstractmethod

class YOLOPOSEBASE(ABC):
    """
    YOLO pose detection model class for handling inference and visualization.

    This class provides functionality to load a YOLO pose model, perform inference on images,
    and visualize the detection results.

    Attributes:
        model_path (str): Path to the ONNX model file.
        input_image (str): Path to the input image file.
        confidence_thres (float): Confidence threshold for filtering detections.
        iou_thres (float): IoU threshold for non-maximum suppression.
        classes (List[str]): List of class names from the COCO pose dataset.
        kpt_shape (Tuple[int, int]): Shape of the keypoints in the model output.
        KPS_COLORS (List[Tuple[int, int, int]]): Colors for keypoints.
        SKELETON (List[Tuple[int, int]]): Skeleton connections for pose visualization.
        LIMB_COLORS (List[Tuple[int, int, int]]): Colors for skeleton limbs.
        
        input_width (int): Width dimension of the model input.
        input_height (int): Height dimension of the model input.
        img (np.ndarray): The loaded input image.
        img_height (int): Height of the input image.
        img_width (int): Width of the input image.
    """

    def __init__(self, model_path: str,  confidence_thres: float, iou_thres: float):
        """
        Initialize an instance of the YOLO pose class.

        Args:
            model_path (str): Path to the model.
            confidence_thres (float): Confidence threshold for filtering detections.
            iou_thres (float): IoU threshold for non-maximum suppression.
        """
        self.model_path = model_path
        self.confidence_thres = confidence_thres
        self.iou_thres = iou_thres

        # Load the class names from the COCO dataset
        yaml_cfg = yaml_load(check_yaml("coco8-pose.yaml"))
        self.classes = yaml_cfg["names"]
        self.kpt_shape = yaml_cfg["kpt_shape"]

        self.KPS_COLORS = [(0, 255, 0),    (0, 255, 0),    (0, 255, 0),    (0, 255, 0),
            (0, 255, 0),    (255, 128, 0),  (255, 128, 0),  (255, 128, 0),  (255, 128, 0),
            (255, 128, 0),  (255, 128, 0),  (51, 153, 255), (51, 153, 255), (51, 153, 255),
            (51, 153, 255), (51, 153, 255), (51, 153, 255), ]

        self.SKELETON = [(16, 14),(14, 12),(17, 15),(15, 13),(12, 13),
            (6, 12),(7, 13),(6, 7),(6, 8),(7, 9),(8, 10),(9, 11),
            (2, 3),(1, 2),(1, 3),(2, 4),(3, 5),(4, 6),(5, 7),]

        self.LIMB_COLORS = [(51, 153, 255), (51, 153, 255), (51, 153, 255), (51, 153, 255),
            (255, 51, 255), (255, 51, 255), (255, 51, 255), (255, 128, 0),  (255, 128, 0),  
            (255, 128, 0),  (255, 128, 0),  (255, 128, 0),  (0, 255, 0),    (0, 255, 0),    
            (0, 255, 0),    (0, 255, 0),    (0, 255, 0),    (0, 255, 0),    (0, 255, 0), ]
        
        self.load_model()

    def letterbox(self, img: np.ndarray, new_shape: Tuple[int, int] = (640, 640)) -> Tuple[np.ndarray, Tuple[int, int]]:
        """
        Resize and reshape images while maintaining aspect ratio by adding padding.

        Args:
            img (np.ndarray): Input image to be resized.
            new_shape (Tuple[int, int]): Target shape (height, width) for the image.

        Returns:
            (np.ndarray): Resized and padded image.
            (Tuple[int, int]): Padding values (top, left) applied to the image.
        """
        shape = img.shape[:2]  # current shape [height, width]

        # Scale ratio (new / old)
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])

        # Compute padding
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = (new_shape[1] - new_unpad[0]) / 2, (new_shape[0] - new_unpad[1]) / 2  # wh padding

        if shape[::-1] != new_unpad:  # resize
            img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))

        return img, (top, left)

    def draw_poses(self, img: np.ndarray, boxes: np.ndarray, scores: np.ndarray, kpts: np.ndarray) -> np.ndarray:
        """
        Draw skeleton and keypoints on the input image based on pose detection results.

        Args:
            img (np.ndarray): Input image.
            scores (np.ndarray): Confidence scores for each detected pose.
            boxes (np.ndarray): Bounding boxes in (x, y, width, height) format.
            kpts (np.ndarray): Detected keypoints for each pose.

        Note:
            `kpts` should be a NumPy array of shape (num_poses, num_keypoints * 3), where each row contains
            (x, y, score) for each keypoint in the order specified by your implementation.
        Returns:
            (np.ndarray): The input image with  bounding boxes, keypoints drawn on it.
        """
        for idx in range(len(scores)):
            score = f"{scores[idx]:.2f}"
            x = int(np.clip(boxes[idx][0] - boxes[idx][2] / 2, 0, img.shape[1]))
            y = int(np.clip(boxes[idx][1] - boxes[idx][3] / 2, 0, img.shape[0]))
            x_plus_w = int(np.clip(x + boxes[idx][2], 0, img.shape[1]))
            y_plus_h = int(np.clip(y + boxes[idx][3], 0, img.shape[0]))

            # Draw bounding box
            cv2.rectangle(img, (x, y), (x_plus_w, y_plus_h), (0, 255, 0), 2)
            cv2.putText(img, score, (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            for k in range(self.kpt_shape[0] + 2):
                if k < self.kpt_shape[0]:
                    kps_x = int(kpts[idx][k * 3 + 0])
                    kps_y = int(kpts[idx][k * 3 + 1])
                    kps_s = kpts[idx][k * 3 + 2]

                    if kps_s > 0.5:
                        kps_color = self.KPS_COLORS[k]
                        cv2.circle(img, (kps_x, kps_y), 5, kps_color, -1)

                    ske = self.SKELETON[k]
                    pos1_x = int(kpts[idx][(ske[0] - 1) * 3])
                    pos1_y = int(kpts[idx][(ske[0] - 1) * 3 + 1])

                    pos2_x = int(kpts[idx][(ske[1] - 1) * 3])
                    pos2_y = int(kpts[idx][(ske[1] - 1) * 3 + 1])

                    pos1_s = int(kpts[idx][(ske[0] - 1) * 3 + 2])
                    pos2_s = int(kpts[idx][(ske[1] - 1) * 3 + 2])

                    if pos1_s > 0.5 and pos2_s > 0.5:
                        limb_color = self.LIMB_COLORS[k]
                        cv2.line(img, (pos1_x, pos1_y), (pos2_x, pos2_y), limb_color, 2)

        return img
    
    def preprocess(self, input_img: np.ndarray) -> Tuple[np.ndarray, Tuple[int, int]]:
        """
        Preprocess the input image before performing inference.

        This method reads the input image, converts its color space, applies letterboxing to maintain aspect ratio,
        normalizes pixel values, and prepares the image data for model input.

        Returns:
            (np.ndarray): Preprocessed image data ready for inference with shape (1, 3, height, width).
            (Tuple[int, int]): Padding values (top, left) applied during letterboxing.
        """
        
        # Get the height and width of the input image
        self.img_height, self.img_width = input_img.shape[:2]

        # Convert the image color space from BGR to RGB
        img = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB)

        img, pad = self.letterbox(img, (self.input_width, self.input_height))

        # Normalize the image data by dividing it by 255.0
        image_data = np.array(img) / 255.0

        # Transpose the image to have the channel dimension as the first dimension
        image_data = np.transpose(image_data, (2, 0, 1))  # Channel first

        # Expand the dimensions of the image data to match the expected input shape
        image_data = np.expand_dims(image_data, axis=0).astype(np.float32)

        # Return the preprocessed image data
        return image_data, pad

    def postprocess(self, output: List[np.ndarray], pad: Tuple[int, int]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Perform post-processing on the model's output to extract and visualize detections.

        This method processes the raw model output to extract bounding boxes, scores, and keypoints.
        It applies non-maximum suppression to filter overlapping detections and draws the results on the input image.

        Args:
            input_image (np.ndarray): The input image.
            output (List[np.ndarray]): The output arrays from the model.
            pad (Tuple[int, int]): Padding values (top, left) used during letterboxing.

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]: The bounding boxes, scores and keypoints.
        """
        gain = min(self.input_height / self.img_height, self.input_width / self.img_height)

        # yolov8 has an output of shape (batchSize, 56,  8400) (17 x point[x,y,prop] + prop + box[x,y,w,h])
        # Transpose and squeeze the output to match the expected shape
        output = np.transpose(np.squeeze(output[0]))

        scores = output[:, 4]
        boxes = output[:, 0:4]
        kpts = output[:, 5:]

        mask = np.array(scores) > self.confidence_thres

        scores = np.array(scores)[mask]
        boxes = np.array(boxes)[mask]
        kpts = np.array(kpts)[mask]
        
        boxes_id = cv2.dnn.NMSBoxes(boxes.tolist(), scores.tolist(), self.confidence_thres, self.iou_thres)

        ret_scores = scores[boxes_id]
        ret_boxes = np.array(boxes[boxes_id])
        ret_kpts = np.array(kpts[boxes_id])
           
        ret_boxes = np.array(boxes[boxes_id]) / gain
        ret_kpts = np.array(kpts[boxes_id]) / gain

        hpad = pad[0]
        wpad = pad[1]
        ret_boxes[:, 0] -= int(wpad/ gain)
        ret_boxes[:, 1] -= int(hpad/ gain)
        ret_kpts[:, 0::3] -= int(wpad/ gain)
        ret_kpts[:, 1::3] -= int(hpad/ gain)

        return (ret_boxes, ret_scores , ret_kpts)
    
    @abstractmethod
    def load_model(self) -> None:
        pass
    
    @abstractmethod
    def inference(self, input_image: np.ndarray) -> np.ndarray:
       pass