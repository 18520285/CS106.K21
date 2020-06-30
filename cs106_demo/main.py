from PyQt5.QtWidgets import QApplication, QPushButton, QDialog, QGroupBox, QLineEdit, QVBoxLayout, QHBoxLayout, QLabel
import sys
from PyQt5 import QtGui
import pickle
from pyvi import ViTokenizer


class Window(QDialog):
    def __init__(self):
        super().__init__()

        self.title = "Sentiment Analysis Vietnamese"
        self.top = 200
        self.left = 500
        self.width = 400
        self.hight = 250
        self.iconName = "uit.png"
        self.InitWindow()

    def InitWindow(self):
        self.setWindowTitle(self.title)
        self.setGeometry(self.left, self.top, self.width, self.hight)
        self.setWindowIcon(QtGui.QIcon(self.iconName))

        self.Createlayout()
        hlayout = QHBoxLayout()
        hlayout.addWidget(self.group)
        self.setLayout(hlayout)

        self.show()

    def Createlayout(self):
        self.group = QGroupBox("Chương trình thực nghiệm")
        vlayout = QVBoxLayout()

        btn1 = QPushButton("Demo", self)
        btn1.setMinimumHeight(50)
        btn1.clicked.connect(self.Open3Dialog)
        vlayout.addWidget(btn1)

        btn2 = QPushButton("Sinh viên thực hiện", self)
        btn2.setMinimumHeight(50)
        btn2.clicked.connect(self.Open4Dialog)
        vlayout.addWidget(btn2)

        self.group.setLayout(vlayout)

    def Open3Dialog(self):
        mydialog = QDialog(self)
        mydialog.setWindowTitle("Demo")
        mydialog.setWindowIcon(QtGui.QIcon(self.iconName))
        mydialog.setGeometry(self.left, self.top, self.width, self.hight)

        group = QGroupBox("Phân tích cảm xúc",mydialog)

        hlayout = QHBoxLayout()

        InGroup = QGroupBox("Input")
        inlayout = QVBoxLayout()
        self.InText =QLineEdit()
        self.InText.setPlaceholderText("Nhập comment cần phần tích")
        inlayout.addWidget(self.InText)
        btn1 = QPushButton("Phân tích")
        btn1.clicked.connect(self.CLassfication)
        inlayout.addWidget(btn1)
        InGroup.setLayout(inlayout)
        hlayout.addWidget(InGroup)

        OutGroup = QGroupBox("Output")
        outlayout = QVBoxLayout()

        self.PosLabel = QLabel("Xác suất pos ")
        self.NegLabel = QLabel("Xác suất neg ")
        self.ClassLabel = QLabel("Kết quả")
        outlayout.addWidget(self.PosLabel)
        outlayout.addWidget(self.NegLabel)
        outlayout.addWidget(self.ClassLabel)
        OutGroup.setLayout(outlayout)
        hlayout.addWidget(OutGroup)

        group.setLayout(hlayout)


        vlayout = QVBoxLayout()
        vlayout.addWidget(group)

        mydialog.setLayout(vlayout)
        mydialog.show()

    def Open4Dialog(self):
        mydialog = QDialog(self)
        mydialog.setWindowTitle("Sinh viên thực hiện")
        mydialog.setWindowIcon(QtGui.QIcon(self.iconName))
        mydialog.setGeometry(self.left, self.top, self.width, self.hight)


        group = QGroupBox("Thông tin sinh viên", mydialog)

        hlayout = QHBoxLayout()

        groupht = QGroupBox("Họ tên")
        groupmssv = QGroupBox("Mã số sinh viên")
        vlayout1 = QVBoxLayout()
        label1 = QLabel("Nguyễn Hữu Hoàng")
        vlayout1.addWidget(label1)
        label2 = QLabel("Phan Phát Huy")
        vlayout1.addWidget(label2)
        label3 = QLabel("Nguyễn Thị Hà")
        vlayout1.addWidget(label3)
        label4 = QLabel("Nguyễn Hải Ngọc")
        vlayout1.addWidget(label4)
        label5 = QLabel("Nguyễn Lê Hoàng Hùng")
        vlayout1.addWidget(label5)
        label6 = QLabel("Nguyễn Võ Hùng Vỹ")
        vlayout1.addWidget(label6)
        groupht.setLayout(vlayout1)

        vlayout2 = QVBoxLayout()
        label1 = QLabel("18520283")
        vlayout2.addWidget(label1)
        label2 = QLabel("18520287")
        vlayout2.addWidget(label2)
        label3 = QLabel("18520691")
        vlayout2.addWidget(label3)
        label4 = QLabel("18520321")
        vlayout2.addWidget(label4)
        label5 = QLabel("18520285")
        vlayout2.addWidget(label5)
        label6 = QLabel("18521683")
        vlayout2.addWidget(label6)
        groupmssv.setLayout(vlayout2)

        hlayout.addWidget(groupht)
        hlayout.addWidget(groupmssv)

        group.setLayout(hlayout)

        vlayout = QVBoxLayout()
        vlayout.addWidget(group)
        mydialog.setLayout(vlayout)

        mydialog.show()

    def CLassfication(self):
        model = pickle.load(open("model.pkl", "rb"))
        vector = pickle.load(open("Countvector.pkl", "rb"))
        s = vector.transform([ViTokenizer.tokenize(self.InText.text())]).toarray()
        if model.predict(s)[0]==1:
            self.ClassLabel.setText("Tích cực")
        else:
            self.ClassLabel.setText("Tiêu cực")
        x, y = model.predict_proba(s)[0]
        self.NegLabel.setText("Probability Negative: %0.3f"%x)
        self.PosLabel.setText("Probability Positive: %0.3f"%y)


if __name__ == "__main__":
    App = QApplication(sys.argv)
    window = Window()
    window.show()
    sys.exit(App.exec())
