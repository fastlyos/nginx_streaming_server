import sys
from PyQt4 import QtCore, QtGui
from PyQt4.phonon import Phonon
app = QtGui.QApplication(sys.argv)
vp = Phonon.VideoPlayer()
vp.show()
media = Phonon.MediaSource('/home/hmcheng/small.mp4')
vp.load(media)
vp.play()
sys.exit(app.exec_())