import torch
import os
from PIL import Image
import matplotlib.pyplot as plt
import pandas as pd
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
    #ax.set_title(self.basename, fontsize=16, color='blue', loc='center')
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
def calculate_result(self):
    transform = T.ToTensor()
    index=0
    for i in self.files:
        image = Image.open(i).convert("RGB")
        image_tensor = transform(image).unsqueeze(0).to(self.device)
        #擷取舟骨位置=>show IMG
        outputs = self.model1(image_tensor)
        output = outputs[0]
        if(len(output["boxes"]))>0:
            pred_boxes = output["boxes"][0].cpu().detach().numpy()  # [x_min, y_min, x_max, y_max]
            part2_img = image.crop((pred_boxes[0], pred_boxes[1], pred_boxes[2], pred_boxes[3]))
            if(index==self.current_file):
                fig, ax = plt.subplots()# 設置繪圖區域
                ax.imshow(part2_img)# 顯示圖片
                ax.axis("off")  # 去除軸標籤
                fig.canvas.manager.set_window_title('part1_image')
                plt.tight_layout()# 使佈局緊湊
                plt.show()  # 顯示圖片
        else:
            return 0
        #判斷骨折
        '''
        results = self.model2(part2_img)  # predict on an image
        pre_box_ori=results.boxes.xyxy[0].tolist()
        x_min, y_min, x_max, y_max = pre_box_ori  # 分別取出座標
        pre_box = [x_min, x_max, y_min, y_max]  # 重新排列為 xxyy 格式
        new_path=i.replace(
            "scaphoid_detection\\images\\",  # 替換的舊部分
            "fracture_detection\\annotations\\"  # 替換為的新部分
        ).replace(".jpg", ".json")  # 將文件擴展名從 .jpg 修改為 .json
        with open(new_path, 'r', encoding='utf-8') as file:
                data = json.load(file)
            
        temp_df = pd.json_normalize(data)# 將 JSON 轉換為 DataFrame
        tar_box=
        '''
        index=index+1


def show_result(self):
    self.label_4.setText("IoU:"+self.Iou)
    self.label_5.setText("Accuracy:"+self.acc)
    self.label_6.setText("Precision:"+self.recicion)
    self.label_7.setText("Recall:"+self.recall)