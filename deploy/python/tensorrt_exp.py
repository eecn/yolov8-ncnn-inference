# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import argparse
import cv2
import numpy as np
import tensorrt as trt

import common
import os

from yolo_pose_base import YOLOPOSEBASE

# https://github.com/NVIDIA/TensorRT/tree/release/10.9/samples/python
TRT_LOGGER = trt.Logger()

# 直接使用ultralytics export出来的模型不能直接使用tensorrt进行推理，进行模型实例化会出错 这里直接使用onnx解析tensorrt模型
def get_engine(onnx_file_path, engine_file_path="",img_height=640,img_width=640):
    """Attempts to load a serialized engine if available, otherwise builds a new TensorRT engine and saves it."""

    def build_engine():
        """Takes an ONNX file and creates a TensorRT engine to run inference with"""
        with trt.Builder(TRT_LOGGER) as builder, builder.create_network(
            0
        ) as network, builder.create_builder_config() as config, trt.OnnxParser(
            network, TRT_LOGGER
        ) as parser, trt.Runtime(
            TRT_LOGGER
        ) as runtime:
            config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 28)  # 256MiB
            # Parse model file
            if not os.path.exists(onnx_file_path):
                print("ONNX file {} not found.".format(onnx_file_path))
                exit(0)
            print("Loading ONNX file from path {}...".format(onnx_file_path))
            with open(onnx_file_path, "rb") as model:
                print("Beginning ONNX file parsing")
                if not parser.parse(model.read()):
                    print("ERROR: Failed to parse the ONNX file.")
                    for error in range(parser.num_errors):
                        print(parser.get_error(error))
                    return None
            network.get_input(0).shape = [1, 3, img_width, img_height]
            print("Completed parsing of ONNX file")
            print( "Building an engine from file {}; this may take a while...".format(onnx_file_path))
            plan = builder.build_serialized_network(network, config)
            engine = runtime.deserialize_cuda_engine(plan)
            print("Completed creating Engine")
            with open(engine_file_path, "wb") as f:
                f.write(plan)
            return engine

    if os.path.exists(engine_file_path):
        # If a serialized engine exists, use it instead of building an engine.
        print("Reading engine from file {}".format(engine_file_path))
        with open(engine_file_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
            return runtime.deserialize_cuda_engine(f.read())
    else:
        return build_engine()

class TENSORRTPose(YOLOPOSEBASE):
    def __init__(self, model_path: str,  confidence_thres: float, iou_thres: float, onnx_path: str=None):
        super(TENSORRTPose, self).__init__(model_path, confidence_thres, iou_thres)
        self.onnx_path = onnx_path

    def load_model(self) -> None:
        #onnx_file_path = "D:\\workspace\\deploy_model\\yolov8-ncnn-inference\\deploy\\python\\yolo11s-pose.onnx"
        #engine_file_path = "D:\\workspace\\deploy_model\\yolov8-ncnn-inference\\deploy\\python\\yolo11s-pose.trt"

        self.model = get_engine(self.model_path, self.engine_file_path, self.input_height, self.input_width)
        self.input_width = 640
        self.input_height = 640
        with open(self.model_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
            engine =  runtime.deserialize_cuda_engine(f.read())
            self.model = engine
        if self.model is None:
            print("Failed to create TensorRT engine.")
            print("Trying to build TensorRT engine from ONNX file instead.")
            self.model = get_engine(self.onnx_path, self.model_path, self.input_height, self.input_width)

       
    def inference(self, input_image: np.ndarray) -> np.ndarray:
        """
        Perform inference using an model and return the boxes, scores and keypoints.

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]: The bounding boxes, scores and keypoints.
        """
        
        # Preprocess the image data
        img_data, pad = self.preprocess(input_image)
        
        context = self.model.create_execution_context()
        inputs, outputs, bindings, stream = common.allocate_buffers(self.model)
        # Set host input to the image. The common.do_inference function will copy the input to the GPU before executing.
        inputs[0].host = np.squeeze(img_data)
        trt_outputs = common.do_inference(
            context,
            engine=self.model,
            bindings=bindings,
            inputs=inputs,
            outputs=outputs,
            stream=stream,
        )
        out = trt_outputs[0].reshape(56,8400)

        outputs = np.expand_dims(np.array(out), axis=0)
        return self.postprocess(outputs, pad)  # output image


if __name__ == "__main__":
    # Create an argument parser to handle command-line arguments
    parser = argparse.ArgumentParser()
    ASSERT = "D://workspace//deploy_model//yolov8-ncnn-inference//deploy//python//"
    parser.add_argument("--model", type=str, default=ASSERT+"yolo11s-pose.trt", help="Input your TensorRT model.")
    parser.add_argument("--img", type=str, default=ASSERT+"bus.jpg", help="Path to input image.")
    parser.add_argument("--conf-thres", type=float, default=0.5, help="Confidence threshold")
    parser.add_argument("--iou-thres", type=float, default=0.5, help="NMS IoU threshold")
    parser.add_argument("--onnx_path", type=str, default=ASSERT+"yolo11s-pose.onnx", help="Path to ONNX model.")
    args = parser.parse_args()

    img = cv2.imread(args.img)


    # Create an instance of the YOLOv8 class with the specified arguments
    detection = TENSORRTPose(args.model, args.conf_thres, args.iou_thres)

    # Perform object detection and obtain the output image
    (ret_boxes, ret_scores , ret_kpts) = detection.inference(img)
    

    output_image = detection.draw_poses(img,ret_boxes, ret_scores , ret_kpts)
    cv2.imwrite(ASSERT+"ret_engine.jpg",output_image)