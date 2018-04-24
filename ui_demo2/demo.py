import os, commands, sys, time, threading
from PyQt4 import QtCore, QtGui, uic
 
video_src_server_ip = "172.18.3.225"
caching_server_ip = "172.18.3.227"

qtCreatorFile = "demo.ui" # Enter file here.
Ui_MainWindow, QtBaseClass = uic.loadUiType(qtCreatorFile)

class MyThread(QtCore.QThread):
    updated = QtCore.pyqtSignal(tuple)

    def run( self ):
        while True:
            epchightotal = int(commands.getstatusoutput('sh get_count.sh ' + caching_server_ip + ' high')[1])
            epclowtotal = int(commands.getstatusoutput('sh get_count.sh ' + caching_server_ip + ' low')[1])
            print "epc: ", epchightotal, epclowtotal
            #self.progressBar_epc.setValue((float(epchightotal)/float(epchightotal+epclowtotal))*100.0)

            remotehightotal = int(commands.getstatusoutput('sh get_count.sh ' + video_src_server_ip + ' high')[1])
            remotelowtotal = int(commands.getstatusoutput('sh get_count.sh ' + video_src_server_ip + ' low')[1])
            print "remote: ", remotehightotal, remotelowtotal
            #self.progressBar_noepc.setValue((float(remotehightotal)/float(remotehightotal+remotelowtotal))*100.0)            
            time.sleep(0.5)

            self.updated.emit(((float(epchightotal)/float(epchightotal+epclowtotal+0.001))*100.0,
             (float(remotehightotal)/float(remotehightotal+remotelowtotal+0.001))*100.0))

class MyApp(QtGui.QMainWindow, Ui_MainWindow):
    def __init__(self):
        QtGui.QMainWindow.__init__(self)
        Ui_MainWindow.__init__(self)
        self.setupUi(self)

        self._thread = MyThread(self)
        self._thread.updated.connect(self.update_stats)
        self._thread.start()

        #start a isolated thread to update the percentage bar solely
        #t = threading.Thread(target=self.update_stats_bar)
        #t.start()

    def update_stats( self, value ):
        self.progressBar_epc.setValue(value[0])
        self.progressBar_noepc.setValue(value[1])
        
 
if __name__ == "__main__":

    #clean up all the previous logs
    os.system("sudo find /var/lib/docker/containers -name *-json.log -delete")

    #os.system("export ZZIP=" +caching_server_ip + "; sudo -E docker-compose -f docker-compose-gui.yml up --no-deps -d --scale app=1")
    #os.system("export ZZIP=" +video_src_server_ip+ "; sudo -E docker-compose -f docker-compose-gui.yml up --no-deps -d --scale app=1")
    os.system("sudo docker-compose -f docker-compose-gui.yml up -d --scale app-gui=2")
    os.system("sudo docker-compose -f docker-compose-gui-epc.yml up -d --scale app-gui-epc=2")

    app = QtGui.QApplication(sys.argv)
    window = MyApp()
    window.show()
    sys.exit(app.exec_())
