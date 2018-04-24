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

qtCreatorFile = "demo_2b.ui" # Enter file here.
Ui_MainWindow, QtBaseClass = uic.loadUiType(qtCreatorFile)

class Player(QWidget):
    """A simple Media Player using VLC and Qt
    """
    def __init__(self, master=None):
        QWidget.__init__(self, master)
        #self.setWindowTitle("Media Player")

        # creating a basic vlc instance
        self.instance = vlc.Instance()
        # creating an empty vlc media player
        self.mediaplayer = self.instance.media_player_new()

        self.createUI()
        self.isPaused = False

    def createUI(self):
        """Set up the user interface, signals & slots
        """
        #self.widget = QWidget(self)
        #self.setCentralWidget(self.widget)

        # In this widget, the video will be drawn
        self.videoframe = QFrame()
        self.palette = self.videoframe.palette()
        self.palette.setColor (QPalette.Window,
                               QColor(0,0,0))
        self.videoframe.setPalette(self.palette)
        self.videoframe.setAutoFillBackground(True)

        # self.positionslider = QSlider(Qt.Horizontal, self)
        # self.positionslider.setToolTip("Position")
        # self.positionslider.setMaximum(1000)
        # self.positionslider.sliderMoved.connect(self.setPosition)

        # self.hbuttonbox = QHBoxLayout()
        # self.playbutton = QPushButton("Play")
        # self.hbuttonbox.addWidget(self.playbutton)
        # self.playbutton.clicked.connect(self.PlayPause)

        # self.stopbutton = QPushButton("Stop")
        # self.hbuttonbox.addWidget(self.stopbutton)
        # self.stopbutton.clicked.connect(self.Stop)

        # self.hbuttonbox.addStretch(1)
        # self.volumeslider = QSlider(Qt.Horizontal, self)
        # self.volumeslider.setMaximum(100)
        # self.volumeslider.setValue(self.mediaplayer.audio_get_volume())
        # self.volumeslider.setToolTip("Volume")
        # self.hbuttonbox.addWidget(self.volumeslider)
        # self.volumeslider.valueChanged.connect(self.setVolume)

        self.vboxlayout = QVBoxLayout()
        self.vboxlayout.addWidget(self.videoframe)
        #self.vboxlayout.addWidget(self.positionslider)
        #self.vboxlayout.addLayout(self.hbuttonbox)

        self.setLayout(self.vboxlayout)

        # open = QAction("&Open", self)
        # open.triggered.connect(self.OpenFile)
        # exit = QAction("&Exit", self)
        # exit.triggered.connect(sys.exit)
        #menubar = self.menuBar()
        #filemenu = menubar.addMenu("&File")
        #filemenu.addAction(open)
        #filemenu.addSeparator()
        #filemenu.addAction(exit)

        self.timer = QTimer(self)
        self.timer.setInterval(200)
        self.timer.timeout.connect(self.updateUI)

    def PlayPause(self):
        """Toggle play/pause status
        """
        if self.mediaplayer.is_playing():
            self.mediaplayer.pause()
            #self.playbutton.setText("Play")
            self.isPaused = True
        else:
            if self.mediaplayer.play() == -1:
                self.OpenFile()
                return
            self.mediaplayer.play()
            #self.playbutton.setText("Pause")
            self.timer.start()
            self.isPaused = False

    def Stop(self):
        """Stop player
        """
        self.mediaplayer.stop()
        #self.playbutton.setText("Play")

    def OpenFile(self, filename=None):
        """Open a media file in a MediaPlayer
        """
        if filename is None:
            filename = QFileDialog.getOpenFileName(self, "Open File", os.path.expanduser('~'))[0]
        if not filename:
            return

        # create the media
        if sys.version < '3':
            filename = unicode(filename)
        self.media = self.instance.media_new(filename)
        # put the media in the media player
        self.mediaplayer.set_media(self.media)

        # parse the metadata of the file
        self.media.parse()
        # set the title of the track as window title
        self.setWindowTitle(self.media.get_meta(0))

        # the media player has to be 'connected' to the QFrame
        # (otherwise a video would be displayed in it's own window)
        # this is platform specific!
        # you have to give the id of the QFrame (or similar object) to
        # vlc, different platforms have different functions for this
        if sys.platform.startswith('linux'): # for Linux using the X Server
            self.mediaplayer.set_xwindow(self.videoframe.winId())
        elif sys.platform == "win32": # for Windows
            self.mediaplayer.set_hwnd(self.videoframe.winId())
        elif sys.platform == "darwin": # for MacOS
            self.mediaplayer.set_nsobject(int(self.videoframe.winId()))
        self.PlayPause()

    def setVolume(self, Volume):
        """Set the volume
        """
        self.mediaplayer.audio_set_volume(Volume)

    def setPosition(self, position):
        """Set the position
        """
        # setting the position to where the slider was dragged
        self.mediaplayer.set_position(position / 1000.0)
        # the vlc MediaPlayer needs a float value between 0 and 1, Qt
        # uses integer variables, so you need a factor; the higher the
        # factor, the more precise are the results
        # (1000 should be enough)

    def updateUI(self):
        """updates the user interface"""
        # setting the slider to the desired position
        #self.positionslider.setValue(self.mediaplayer.get_position() * 1000)

        if not self.mediaplayer.is_playing():
            # no need to call this function if nothing is played
            self.timer.stop()
            if not self.isPaused:
                # after the video finished, the play button stills shows
                # "Pause", not the desired behavior of a media player
                # this will fix it
                self.Stop()


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

        self.resetbandwidth_button.clicked.connect(self.ReSetBandwidth)
        self.setbandwidth_button.clicked.connect(self.SetBandwidth)

        self._thread = MyThread(self)
        self._thread.updated.connect(self.update_stats)
        self._thread.start()

        player = Player()
        player.OpenFile("http://" + caching_server_ip + "/hls/sample9.m3u8")
        #player.resize(512,288)
        player.PlayPause()
        self.grid.addWidget(player)


    def update_stats( self, value ):
        self.bandwidth.setText(str(value*(1+vm_scale)))

        
    def closeEvent(self, event):
        #os.system("sudo docker kill $(sudo docker ps -a -q)")
        subprocess.call("sudo docker kill $(sudo docker ps -a -q)", shell=True)
        subprocess.call("ssh hmcheng@" + caching_server_ip + " 'echo h0940232|sudo -S wondershaper -a eno1 -c'", shell=True)

    def SetBandwidth(self):
        print ("set video src upload bandwidth")
        subprocess.call("ssh hmcheng@" + caching_server_ip + " 'echo h0940232|sudo -S wondershaper -a eno1 -c'", shell=True)
        subprocess.call("ssh hmcheng@" + caching_server_ip + " 'sleep 0.1s'", shell=True)
        subprocess.call("ssh hmcheng@" + caching_server_ip + " 'echo h0940232|sudo -S wondershaper -a eno1 -u " + self.setbandwidth.text() +"'", shell=True)

        msg = QMessageBox()
        msg.setIcon(QMessageBox.Information)

        msg.setText("MEC caching server max bandwidth is set!")
        msg.setWindowTitle("Information")
        msg.setStandardButtons(QMessageBox.Ok)
        msg.exec_()

    def ReSetBandwidth(self):
        print ("reset video src upload bandwidth")
        subprocess.call("ssh hmcheng@" + caching_server_ip + " 'echo h0940232|sudo -S wondershaper -a eno1 -c'", shell=True)

        msg = QMessageBox()
        msg.setIcon(QMessageBox.Information)

        msg.setText("MEC caching  max bandwidth is reset!")
        msg.setWindowTitle("Information")
        msg.setStandardButtons(QMessageBox.Ok)
        msg.exec_()

if __name__ == "__main__":

    #clean up all the previous logs and reset video server bandwidth
    subprocess.call("sudo docker kill $(sudo docker ps -a -q)", shell=True)
    subprocess.call("ssh hmcheng@" + caching_server_ip + " 'echo h0940232|sudo -S wondershaper -a eno1 -c'", shell=True)
    #subprocess.call("sudo find /var/lib/docker/containers -name *-json.log -delete", shell=True)

    #os.system("sudo docker-compose -f ../docker-compose.yml up -d --scale app=" + str(vm_scale))
    subprocess.call("sudo docker-compose -f ../docker-compose-epc.yml up -d --scale app-epc=" + str(vm_scale), shell=True)

    #give docker some time to load the vm
    #time.sleep(3)


    app = QApplication(sys.argv)
    locale.setlocale(locale.LC_NUMERIC, 'C')
    window = MyApp()
    window.show()
    sys.exit(app.exec_())
