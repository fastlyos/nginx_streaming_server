import os, commands, sys, time, threading
from PyQt5 import uic
from PyQt5.QtCore import Qt, QTimer, QThread, pyqtSignal
from PyQt5.QtGui import QPalette, QColor
from PyQt5.QtWidgets import QMainWindow, QWidget, QFrame, QSlider, QHBoxLayout, QPushButton, \
    QVBoxLayout, QAction, QFileDialog, QApplication, QMessageBox, QLabel
import vlc
 
video_src_server_ip = "172.18.3.225"
caching_server_ip = "172.18.3.227"
vm_scale = 7
num_of_player = 3

qtCreatorFile = "demo.ui" # Enter file here.
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

        self.positionslider = QSlider(Qt.Horizontal, self)
        self.positionslider.setToolTip("Position")
        self.positionslider.setMaximum(1000)
        self.positionslider.sliderMoved.connect(self.setPosition)

        self.hbuttonbox = QHBoxLayout()
        self.playbutton = QPushButton("Play")
        self.hbuttonbox.addWidget(self.playbutton)
        self.playbutton.clicked.connect(self.PlayPause)

        self.stopbutton = QPushButton("Stop")
        self.hbuttonbox.addWidget(self.stopbutton)
        self.stopbutton.clicked.connect(self.Stop)

        self.hbuttonbox.addStretch(1)
        self.volumeslider = QSlider(Qt.Horizontal, self)
        self.volumeslider.setMaximum(100)
        self.volumeslider.setValue(self.mediaplayer.audio_get_volume())
        self.volumeslider.setToolTip("Volume")
        self.hbuttonbox.addWidget(self.volumeslider)
        self.volumeslider.valueChanged.connect(self.setVolume)

        self.vboxlayout = QVBoxLayout()
        self.vboxlayout.addWidget(self.videoframe)
        self.vboxlayout.addWidget(self.positionslider)
        self.vboxlayout.addLayout(self.hbuttonbox)

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
            self.playbutton.setText("Play")
            self.isPaused = True
        else:
            if self.mediaplayer.play() == -1:
                self.OpenFile()
                return
            self.mediaplayer.play()
            self.playbutton.setText("Pause")
            self.timer.start()
            self.isPaused = False

    def Stop(self):
        """Stop player
        """
        self.mediaplayer.stop()
        self.playbutton.setText("Play")

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
        self.positionslider.setValue(self.mediaplayer.get_position() * 1000)

        if not self.mediaplayer.is_playing():
            # no need to call this function if nothing is played
            self.timer.stop()
            if not self.isPaused:
                # after the video finished, the play button stills shows
                # "Pause", not the desired behavior of a media player
                # this will fix it
                self.Stop()


#tuple = [epc%, no_epc%, epc_client, no_epc client, bandwidth]
class MyThread(QThread):
    updated = pyqtSignal(tuple)

    def run( self ):
        while True:
            epchightotal = int(commands.getstatusoutput('sh ../get_count.sh ' + caching_server_ip + ' high')[1])
            epclowtotal = int(commands.getstatusoutput('sh ../get_count.sh ' + caching_server_ip + ' low')[1])
            #print "epc: ", epchightotal, epclowtotal
            #self.progressBar_epc.setValue((float(epchightotal)/float(epchightotal+epclowtotal))*100.0)

            remotehightotal = int(commands.getstatusoutput('sh ../get_count.sh ' + video_src_server_ip + ' high')[1])
            remotelowtotal = int(commands.getstatusoutput('sh ../get_count.sh ' + video_src_server_ip + ' low')[1])
            #print "remote: ", remotehightotal, remotelowtotal
            #self.progressBar_noepc.setValue((float(remotehightotal)/float(remotehightotal+remotelowtotal))*100.0)      
            
            epc_client_num =  int(commands.getstatusoutput('sh ../client_count.sh ' + caching_server_ip)[1])
            noepc_client_num =  int(commands.getstatusoutput('sh ../client_count.sh ' + video_src_server_ip)[1])

            try:
                bandwidth = int(commands.getstatusoutput('sh ../get_bandwidth.sh ' + video_src_server_ip)[1])/1024
            except:
                bandwidth = 0
                  
            time.sleep(0.5)

            #use double brackets since it is a tuple
            self.updated.emit(((float(epchightotal)/float(epchightotal+epclowtotal+0.001))*100.0,
             (float(remotehightotal)/float(remotehightotal+remotelowtotal+0.001))*100.0,
             epc_client_num, noepc_client_num, bandwidth))



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

        #start a isolated thread to update the percentage bar solely
        #t = threading.Thread(target=self.update_stats_bar)
        #t.start()

        #launch the players in ivdeo src server and caching server
        for i in range(0,num_of_player):
            bin_of_i = bin(i+8)
            player = Player()
            time.sleep(1)
            player.OpenFile("http://" + video_src_server_ip + "/hls/sample9.m3u8")
            #player.show()
            player.resize(512,288)
            player.PlayPause()
            self.grid_noepc.addWidget(player,int(bin_of_i[-1]),int(bin_of_i[-2]))

        for i in range(0,num_of_player):
            bin_of_i = bin(i+8)
            player = Player()
            time.sleep(1)
            player.OpenFile("http://" + caching_server_ip + "/hls/sample9.m3u8")
            #player.show()
            player.resize(512,288)
            player.PlayPause()
            self.grid_epc.addWidget(player,int(bin_of_i[-1]),int(bin_of_i[-2]))
        
        # add a block of description
        if num_of_player<4:
            epc_desc = QLabel()
            epc_desc.setText("<html><head/><body><span style=' font-size:10pt; font-weight:600; color:#ff0000;'><p>VLC player clients get the video via the </p><p>caching server (inside MEC)</p>The video quality is much higher since the </p><p>EU has larger bandwidth connecting to the EPC</p></span></body></html>")
            self.grid_epc.addWidget(epc_desc,1, 1)

            noepc_desc = QLabel()
            noepc_desc.setText("<html><head/><body><span style=' font-size:10pt; font-weight:600; color:#ff0000;'><p>VLC player clients get the video directly from </p><p>the video source server (i.e. a remote server)</p><p>The video quality is unstable and swap resolution </p><p>dynamically according to the current bandwidth</p><p>Try this out by setting the video server max </p><p>bandwidth above!</p></span></body></html>")
            self.grid_noepc.addWidget(noepc_desc,1, 1)

    def update_stats( self, value ):
        self.progressBar_epc.setValue(value[0])
        self.progressBar_noepc.setValue(value[1])
        self.client_num_epc.setText(str(value[2]+num_of_player)) #since 4 more clients on the pyqt
        self.client_num_noepc.setText(str(value[3]+num_of_player))  #since 4 more clients on the pyqt
        self.bandwidth.setText(str(value[4]))
        self.bandwidth_2.setText(str(value[4]))
        
    def closeEvent(self, event):
        os.system("sudo docker kill $(sudo docker ps -a -q)")

    def SetBandwidth(self):
        print "set video src upload bandwidth"
        os.system("ssh hmcheng@" + video_src_server_ip + " 'echo h0940232|sudo -S wondershaper -a eno1 -c'")
        os.system("ssh hmcheng@" + video_src_server_ip + " 'sleep 0.1s'")
        os.system("ssh hmcheng@" + video_src_server_ip + " 'echo h0940232|sudo -S wondershaper -a eno1 -u " + self.setbandwidth.text() +"'")

        msg = QMessageBox()
        msg.setIcon(QMessageBox.Information)

        msg.setText("Video server max bandwidth is set!")
        msg.setWindowTitle("Information")
        msg.setStandardButtons(QMessageBox.Ok)
        msg.exec_()

    def ReSetBandwidth(self):
        print "reset video src upload bandwidth"
        os.system("ssh hmcheng@" + video_src_server_ip + " 'echo h0940232|sudo -S wondershaper -a eno1 -c'")

        msg = QMessageBox()
        msg.setIcon(QMessageBox.Information)

        msg.setText("Video server max bandwidth is reset!")
        msg.setWindowTitle("Information")
        msg.setStandardButtons(QMessageBox.Ok)
        msg.exec_()

if __name__ == "__main__":

    #clean up all the previous logs and reset video server bandwidth
    os.system("sudo docker kill $(sudo docker ps -a -q)")
    os.system("ssh hmcheng@" + video_src_server_ip + " 'echo h0940232|sudo -S wondershaper -a eno1 -c'")
    os.system("sudo find /var/lib/docker/containers -name *-json.log -delete")
    # os.system("sudo docker-compose -f docker-compose-gui.yml up -d --scale app-gui=2")
    # os.system("sudo docker-compose -f docker-compose-gui-epc.yml up -d --scale app-gui-epc=2")

    os.system("sudo docker-compose -f ../docker-compose.yml up -d --scale app=" + str(vm_scale))
    os.system("sudo docker-compose -f ../docker-compose-epc.yml up -d --scale app-epc=" + str(vm_scale))

    #give docker some time to load the vm
    #time.sleep(3)

    app = QApplication(sys.argv)
    window = MyApp()
    window.show()
    sys.exit(app.exec_())
