import sys
from PyQt5.QtWidgets import QMainWindow,QApplication,QWidget
from ui.Ui_main import Ui_Dialog
from get_response import split_and_generate

class MyMainWindow(QMainWindow,Ui_Dialog): 
    def __init__(self,parent =None):
        super(MyMainWindow,self).__init__(parent)
        self.setupUi(self)
        self.radioButton.setChecked(True)
        self.pushButton.clicked.connect(self.get_result)

    def get_result(self):
        if self.radioButton.isChecked():
            a=1
        else:
            a=0 
        res=split_and_generate(self.textEdit_2.toPlainText(),a)
        self.textEdit.setText(res) 
 
if __name__ == "__main__":
    app = QApplication(sys.argv)
    myWin = MyMainWindow()
    myWin.show()
    sys.exit(app.exec_())    