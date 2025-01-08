import torch
import os
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import json
from ultralytics import YOLO
from torchvision import transforms as T


def initial(self):
    self.device=torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    self.model1=torch.load("./model1.pth", map_location=self.device)
    self.model2= YOLO("./best.pt")
    
def show_image(self):
    self.basename=os.path.basename(self.files[self.current_file])
    self.label_3.setText("Current Image:"+self.basename)
    image = Image.open(self.files[self.current_file]).convert("RGB")
    fig, ax = plt.subplots()# 設置繪圖區域
    ax.imshow(image)# 顯示圖片
    ax.axis("off")  # 去除軸標籤
    fig.canvas.manager.set_window_title(self.basename)
    plt.tight_layout()# 使佈局緊湊
    plt.show()  # 顯示圖片


def get_ans(self):
    print('enter_get_ans')
    answer=pd.DataFrame(columns=["basename","answer"]) #basename存image name
    index=0
    for i in self.files:
        print('i')
        basename=os.path.basename(i)
        new_path = i.replace("scaphoid_detection/images", "fracture_detection/annotations")
        new_path = new_path.replace(".jpg", ".json")
        print(new_path)
        with open(new_path, 'r', encoding='utf-8') as file:
                data = json.load(file)            
        temp_df = pd.json_normalize(data)# 將 JSON 轉換為 DataFrame
        if(temp_df["name"].iloc[0]=="Fracture"):
            fracture_check=1
        else:
            fracture_check=0
        answer=answer._append({'basename':basename,'answer':fracture_check},ignore_index=True)
    print("answer")
    self.answer=answer.copy()
    print(self.answer)

#def calculate_iou(pre_box,target_box):
    #確定
    
def calculate_one_result(self):
    transform = T.ToTensor()
    image = Image.open(self.files[self.current_file]).convert("RGB")
    image_tensor = transform(image).unsqueeze(0).to(self.device)
    #擷取舟骨位置=>show IMG
    outputs = self.model1(image_tensor)
    output = outputs[0]
    if(len(output["boxes"]))>0:
        pred_boxes = output["boxes"][0].cpu().detach().numpy()  # [x_min, y_min, x_max, y_max]
        part1_img = image.crop((pred_boxes[0], pred_boxes[1], pred_boxes[2], pred_boxes[3]))
    else:
        return 0
    
    #判斷骨折=>show IMG 
    results = self.model2(part1_img, conf=0.25)  # Predict on an image
    part1_img_np = np.array(part1_img)
    part1_img_np_ori = np.array(part1_img)
    for result in results:  # 遍歷每張影像的檢測結果
        if result.obb is not None and len(result.obb) > 0:  # 檢查是否有檢測結果     
            for box in result.obb.xyxyxyxy:  # 迭代每个检测框的多边形
                points = np.array(box.tolist(), dtype=np.int32).reshape((-1, 1, 2))# 转换为 NumPy 数组并调整形状
                cv2.polylines(part1_img_np, [points], isClosed=True, color=(255,255, 255), thickness=2)# 使用 cv2.polylines 绘制封闭的多边形

            # 使用 Matplotlib 显示图像
            plt.figure()
            plt.imshow(cv2.cvtColor(part1_img_np, cv2.COLOR_BGR2RGB))  # 转换为 RGB 格式以供显示
            plt.title('Fracture Detection')
            plt.axis("off")  # 隐藏坐标轴
            plt.show()
        else:
            plt.figure()
            plt.imshow(cv2.cvtColor(part1_img_np_ori, cv2.COLOR_BGR2RGB))  # 转换为 RGB 格式以供显示
            plt.title('Fracture Detection')
            plt.axis("off")  # 隐藏坐标轴
            plt.show()
            return 1
        '''
        results = self.model2(part2_img)  # predict on an image
        pre_box_ori=results.boxes.xyxy[0].tolist()
        x_min, y_min, x_max, y_max = pre_box_ori  # 分別取出座標
        pre_box = [x_min, x_max, y_min, y_max]  # 重新排列為 xxyy 格式
        new_path = i.replace("scaphoid_detection/images", "fracture_detection/annotations")
        new_path = new_path.replace(".jpg", ".json")
        with open(new_path, 'r', encoding='utf-8') as file:
                data = json.load(file)
            
        temp_df = pd.json_normalize(data)# 將 JSON 轉換為 DataFrame
        tar_box=
        '''


def show_result(self):
    self.label_4.setText("IoU:"+self.Iou)
    self.label_5.setText("Accuracy:"+self.acc)
    self.label_6.setText("Precision:"+self.recicion)
    self.label_7.setText("Recall:"+self.recall)