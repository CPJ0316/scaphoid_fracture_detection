import sys
import os
from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog, QMessageBox
from PyQt5.uic import loadUi
from PIL import Image
import HW2
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

class Main_UI(QMainWindow):
    def __init__(self, parent = None):
        super(Main_UI,self).__init__(parent)
        loadUi("./HW2.ui",self)
        self.device=None
        self.model1=None
        self.model2=None
        self.IoU=0
        self.acc=0
        self.precision=0
        self.recall=0
        self.files=[]
        self.answer=[]
        self.predict=[]
        self.current_file=0
        self.basename=""
        HW2.initial(self)
        self.Connect_btn()


    def Connect_btn(self):
        self.pushButton.clicked.connect(self.pushButton1F)  #load file         
        self.pushButton_2.clicked.connect(self.pushButton2F) #Pre        
        self.pushButton_3.clicked.connect(self.pushButton3F) #Next        
        self.pushButton_4.clicked.connect(self.pushButton4F) #Detection 
                  
    def pushButton1F(self):
        folder_path = QFileDialog.getExistingDirectory(self, "Select Folder")
        if not folder_path:
            QMessageBox.warning(self, "操作錯誤提示", "請重新選擇資料夾")
        else:
            self.loadFiles = folder_path  
            print("Selected folder:", self.loadFiles)  
            for file_name in sorted(os.listdir(self.loadFiles)):
                if file_name.endswith(".jpg"):
                    self.files.append(os.path.join(self.loadFiles, file_name))
            self.current_file=0
            print('get_ans')
            HW2.get_ans(self)
            print('show_img')
            HW2.show_image(self)
    def pushButton2F(self):
        if(self.current_file==0):
            QMessageBox.warning(self, "操作錯誤提示", "已經是第1張影像")
        else:
            self.current_file=self.current_file-1
            HW2.show_image(self)
    def pushButton3F(self):
        if(self.current_file==len(self.files)-1):
            QMessageBox.warning(self, "操作錯誤提示", "已經是最後1張影像")
        else:
            self.current_file=self.current_file+1
            HW2.show_image(self)
    def pushButton4F(self):
        error=HW2.calculate_result(self)
        if(error==0):
            QMessageBox.warning(self, "影像判別異常", "沒有找到舟骨")
        #HW2.show_result(self)
        

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = Main_UI()
    window.show()
    sys.exit(app.exec_())