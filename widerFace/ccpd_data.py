import os
import cv2
import shutil

if __name__=="__main__":
    base_dir = "/home/wangke/Datasets/CCPD2020/ccpd_green"
    dst_dir = "/home/wangke/Datasets/CCPD2020GREEN"

    dataset_lists = ["train","val","test"]
    sub_dir_lists = ["images","labels"]
    for i, sub_dir_list in enumerate(sub_dir_lists):
        for j, dataset_list in enumerate(dataset_lists):
            dst_sub_dir_list = os.path.join(dst_dir, sub_dir_list, dataset_list)
            os.makedirs(dst_sub_dir_list, exist_ok=True)

    train_dir = os.path.join(base_dir,"train")
    test_dir = os.path.join(base_dir,"test")
    val_dir = os.path.join(base_dir,"val")

    dir_list = [train_dir, test_dir, val_dir]
    for dir in dir_list:
        image_list = os.listdir(dir)
        if "train" in dir:
            image_dst_folder = os.path.join(dst_dir,"images","train")
        elif "test" in dir:
            image_dst_folder = os.path.join(dst_dir,"images","test")
        else:
            image_dst_folder = os.path.join(dst_dir,"images","val")

        for image in image_list:
            if not image.endswith(".jpg"):
                continue
            annotations = []
            image_path = os.path.join(dir,image)
            # print(image_path)
            image_height, image_width = cv2.imread(image_path).shape[:-1]
            image_name = os.path.basename(image_path).split('.')[0]
            # label_path = image_path.replace(".jpg",".txt")
             # 根据图像名分割标注
            _, _, box, points, label, brightness, blurriness = image_name.split('-')

            box = box.split('_')
            box = [list(map(int, i.split('&'))) for i in box]
            left,top,right,bottom = box[0][0],box[0][1],box[1][0],box[1][1]
            cx = left + (right-left)//2
            cy = top + (bottom-top)//2
            width = right - left
            height = bottom - top
            cx_percent = round(cx/image_width, 6)
            cy_percent = round(cy/image_height, 6)
            width_percent = round(width/image_width, 6)
            height_percent = round(height/image_width, 6)
            annotations.append(f"0 {cx_percent} {cy_percent} {width_percent} {height_percent}")
                   
            points = points.split('_')
            points = [list(map(int, i.split('&'))) for i in points]
            points = points[-2:]+points[:2] # 将关键点的顺序变为从左上顺时针开始
            
            # # 提取所有x和y坐标
            # x_coords = [point[0] for point in points]
            # y_coords = [point[1] for point in points]

            # # 计算x和y坐标的最小和最大值
            # X_min = min(x_coords)
            # X_max = max(x_coords)
            # Y_min = min(y_coords)
            # Y_max = max(y_coords)

            # # 计算左上角和右下角点
            # top_left = (X_min, Y_min)
            # bottom_right = (X_max, Y_max)

            # # 计算中心点坐标、宽度和高度
            # cx = (X_min + X_max) / 2
            # cy = (Y_min + Y_max) / 2
            # width = X_max - X_min
            # height = Y_max - Y_min

            # cx_percent = round(cx/image_width, 6)
            # cy_percent = round(cy/image_height, 6)
            # width_percent = round(width/image_width, 6)
            # height_percent = round(height/image_width, 6)
            # annotations.append(f"0 {cx_percent} {cy_percent} {width_percent} {height_percent}")

            points_percent = []
            for point in points:
                x_percent = round(point[0]/image_width,6)
                y_percent = round(point[1]/image_height,6)
                annotations.append(f"{x_percent} {y_percent} 2")
            annotations = ' '.join(annotations)
            annotations = annotations+"\n"
            
            shutil.copy(image_path,image_dst_folder)
            label_path = os.path.join(image_dst_folder.replace("images","labels"),image_name+".txt")
            
            with open(label_path,"w") as f:
                f.write(annotations)         
