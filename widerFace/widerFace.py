import os
import cv2
import shutil
import random
import numpy as np

def process_annotations(annotation_file_path, base_path):
    labels_dict = {}
    with open(annotation_file_path, 'r') as file:
        annotation_data = file.read()
    cnt = 0
    labels = []
    pre_image_name = ""
    for line in annotation_data.split('\n'):
        if line == "":
            continue
        # if cnt == 20:
        #     return  labels_dict
        if  line.startswith('#'):
            parts = line.split()
            image_path = parts[1]
            full_image_path = os.path.join(base_path, image_path)
            if pre_image_name == "":
                pre_image_name = full_image_path
            
            if labels != []:   
                labels_dict[pre_image_name] = ' '.join(labels)
                labels=[]
                if line == "# 123":
                    return labels_dict
            # print(full_image_path)
            img = cv2.imread(full_image_path)
            height, width = img.shape[0], img.shape[1]
            pre_image_name = full_image_path
            cnt+=1
            continue

        annotations = line.split()
        label = []
        bbox = annotations[0:4]
        keypoints = annotations[4:19]
        score = annotations[19]
        if -1 in [float(bbox[j]) for j in range(4)]:
            continue
        # if -1 in [float(keypoints[j]) for j in range(3)]:
        #     continue
        x, y, w, h = [float(bbox[j]) for j in range(4)]
        x_percent = round(x / width, 6)
        y_percent = round(y / height, 6)
        w_percent = round(w / width, 6)
        h_percent = round(h / height, 6)

        cx_percent = round((x + w / 2) / width, 6)
        cy_percent = round((y + h / 2) / height, 6)
        w_percent = round(w / width, 6)
        label.append(f"0 {cx_percent} {cy_percent} {w_percent} {h_percent}")
        for k in range(5):
            x, y, v = float(keypoints[3*k+0]), float(keypoints[3*k+1]), keypoints[3*k+2]
            if v == "-1.0":  # 无标注
                label.append("0 0 0")
            else:
                x_percent = round(x / width, 6) if x != "-1.0" else 0
                y_percent = round(y / height, 6) if y != "-1.0" else 0
                label.append(f"{x_percent} {y_percent} 2")
        label = ' '.join(label)
        label = label+"\n"
        labels.append(label)

    return labels_dict

if __name__ == '__main__':
  
    base_path = '/mnt/d/workspace/github/datasets/widerface/WIDER_train/WIDER_train/images' # 图像目录
    dst_img_path = '/mnt/d/workspace/github/datasets/widerface/WIDER_train/images'  # 目标图像目录
    dst_label_path = '/mnt/d/workspace/github/datasets/widerface/WIDER_train/labels' # 目标标签目录
    annotation_file_path = '/mnt/d/workspace/github/datasets/widerface/retinaface_gt_v1.1/train/label.txt' # 标注文件路径
    
    labels_dict = process_annotations(annotation_file_path, base_path)

    ratios=[0.8, 0.1, 0.1]
    cumsum_tarios = np.cumsum(ratios)

    items = list(labels_dict.items())
    random.shuffle(items)
    
    # 计算切分点
    split_points = [int(len(items) * ratio) for ratio in cumsum_tarios[:-1]]
    split_points = [0] + split_points + [len(items)]
    
    # 根据切分点分割列表并创建子字典
    split_dicts = [dict(items[split_points[i]:split_points[i+1]]) for i in range(len(ratios))]
    train_dataset, val_dataset, test_dataset = split_dicts
    
    dst_dir = '/mnt/d/workspace/github/datasets/widerface/WIDER_train/WIDER-face'
    dataset_lists = ["train","val","test"]
    sub_dir_lists = ["images","labels"]
    for i, sub_dir_list in enumerate(sub_dir_lists):
        for j, dataset_list in enumerate(dataset_lists):
            dst_sub_dir_list = os.path.join(dst_dir, sub_dir_list, dataset_list)
            os.makedirs(dst_sub_dir_list, exist_ok=True)
    
    for i, image_file_path in enumerate(train_dataset):
        image_dir,image_name = os.path.split(image_file_path)
        dst_image_path = os.path.join(dst_dir, "images", "train", image_name)
        dst_label_path = os.path.join(dst_dir, "labels", "train", image_name.replace(".jpg", ".txt"))
        shutil.copy(image_file_path, dst_image_path)
        label = train_dataset[image_file_path]
        label = '\n'.join([line.lstrip() for line in label.split('\n')])
        with open(dst_label_path, 'w') as f:
            f.write(label)

    for i, image_file_path in enumerate(val_dataset):
        image_dir,image_name = os.path.split(image_file_path)
        dst_image_path = os.path.join(dst_dir, "images", "val", image_name)
        dst_label_path = os.path.join(dst_dir, "labels", "val", image_name.replace(".jpg", ".txt"))
        shutil.copy(image_file_path, dst_image_path)
        label = val_dataset[image_file_path]
        label = '\n'.join([line.lstrip() for line in label.split('\n')])
        with open(dst_label_path, 'w') as f:
            f.write(label)

    for i, image_file_path in enumerate(test_dataset):
        image_dir,image_name = os.path.split(image_file_path)
        dst_image_path = os.path.join(dst_dir, "images", "test", image_name)
        dst_label_path = os.path.join(dst_dir, "labels", "test", image_name.replace(".jpg", ".txt"))
        shutil.copy(image_file_path, dst_image_path)
        label = test_dataset[image_file_path]
        label = '\n'.join([line.lstrip() for line in label.split('\n')])
        with open(dst_label_path, 'w') as f:
            f.write(label)