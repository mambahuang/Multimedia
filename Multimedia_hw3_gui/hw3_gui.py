# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'hw3_gui.ui'
#
# Created by: PyQt5 UI code generator 5.15.10
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_Dialog(object):
    def setupUi(self, Dialog):
        Dialog.setObjectName("Dialog")
        Dialog.resize(400, 300)
        self.btn_edge = QtWidgets.QPushButton(Dialog)
        self.btn_edge.setGeometry(QtCore.QRect(70, 70, 261, 51))
        font = QtGui.QFont()
        font.setFamily("MV Boli")
        font.setPointSize(9)
        font.setBold(True)
        font.setWeight(75)
        self.btn_edge.setFont(font)
        self.btn_edge.setObjectName("btn_edge")
        self.btn_histEqu = QtWidgets.QPushButton(Dialog)
        self.btn_histEqu.setGeometry(QtCore.QRect(70, 170, 261, 51))
        font = QtGui.QFont()
        font.setFamily("MV Boli")
        font.setPointSize(9)
        font.setBold(True)
        font.setItalic(False)
        font.setWeight(75)
        font.setStrikeOut(False)
        self.btn_histEqu.setFont(font)
        self.btn_histEqu.setObjectName("btn_histEqu")

        self.retranslateUi(Dialog)
        QtCore.QMetaObject.connectSlotsByName(Dialog)

    def retranslateUi(self, Dialog):
        _translate = QtCore.QCoreApplication.translate
        Dialog.setWindowTitle(_translate("Dialog", "Dialog"))
        self.btn_edge.setText(_translate("Dialog", "Edge Detection"))
        self.btn_histEqu.setText(_translate("Dialog", "Histogram Equalization"))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    Dialog = QtWidgets.QDialog()
    ui = Ui_Dialog()
    ui.setupUi(Dialog)
    Dialog.show()
    sys.exit(app.exec_())
