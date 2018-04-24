import os, sys, time, threading, subprocess
from PyQt5 import uic
from PyQt5.QtCore import Qt, QTimer, QThread, pyqtSignal
from PyQt5.QtGui import QPalette, QColor
from PyQt5.QtWidgets import QMainWindow, QWidget, QFrame, QSlider, QHBoxLayout, QPushButton, \
    QVBoxLayout, QAction, QFileDialog, QApplication, QMessageBox, QLabel
import vlc

import mpv
import locale

from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
 
video_src_server_ip = "172.18.3.225"
caching_server_ip = "172.18.3.227"
vm_scale = 1
#num_of_player = 3

qtCreatorFile = "demo_2a.ui" # Enter file here.
Ui_MainWindow, QtBaseClass = uic.loadUiType(qtCreatorFile)

class MyThread(QThread):
    updated = pyqtSignal(int)

    def run( self ):
        while True:
            try:
                bandwidth = int(subprocess.check_output('sh ../get_bandwidth.sh ' + caching_server_ip, shell=True))/1024
            except:
                bandwidth = 0
                  
            time.sleep(0.5)

            #use double brackets since it is a tuple
            self.updated.emit(bandwidth)



class MyApp(QMainWindow, Ui_MainWindow):
    def __init__(self):
        QMainWindow.__init__(self)
        Ui_MainWindow.__init__(self)
        self.setupUi(self)

        self._thread = MyThread(self)
        self._thread.updated.connect(self.update_stats)
        self._thread.start()

        player = mpv.MPV(wid=str(int(self.widget.winId())),
                vo='x11'#, # You may not need this
                #log_handler=print,
                #loglevel='debug'
                )
        player.play('http://' + caching_server_ip + '/hls/sample9.m3u8')
        player.show_progress()

    def update_stats( self, value ):
        self.bandwidth.setText(str(value*(1+vm_scale)))

        
    def closeEvent(self, event):
        #os.system("sudo docker kill $(sudo docker ps -a -q)")
        subprocess.call("sudo docker kill $(sudo docker ps -a -q)", shell=True)

if __name__ == "__main__":

    #clean up all the previous logs and reset video server bandwidth
    subprocess.call("sudo docker kill $(sudo docker ps -a -q)", shell=True)
    subprocess.call("ssh hmcheng@" + caching_server_ip + " 'echo h0940232|sudo -S wondershaper -a eno1 -c'", shell=True)
    subprocess.call("sudo find /var/lib/docker/containers -name *-json.log -delete", shell=True)

    #os.system("sudo docker-compose -f ../docker-compose.yml up -d --scale app=" + str(vm_scale))
    subprocess.call("sudo docker-compose -f ../docker-compose-epc.yml up -d --scale app-epc=" + str(vm_scale), shell=True)

    #give docker some time to load the vm
    #time.sleep(3)


    app = QApplication(sys.argv)
    locale.setlocale(locale.LC_NUMERIC, 'C')
    window = MyApp()
    window.show()
    sys.exit(app.exec_())
