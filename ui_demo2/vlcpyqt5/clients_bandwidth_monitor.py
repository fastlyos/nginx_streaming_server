from pyqtgraph.Qt import QtGui, QtCore
import numpy as np
import pyqtgraph as pg
import subprocess, time

video_src_server_ip = "172.18.3.225"
caching_server_ip = "172.18.3.227"

#1st layer: different client, 2nd layer: history bandwidth
bandwidth = []
curve = []
max_history_second =30
update_freq=1000 # in ms

def AddNewBandwidth():
    global bandwidth
    all_bandwidth = subprocess.check_output('sh ../get_all_bandwidth.sh ' + caching_server_ip, shell=True)
    splited = all_bandwidth.splitlines()
    splited_int = map(int, splited)
    splited_int = [x/1024 for x in splited_int]
    #splited_int = [x/1024 for x in splited_int]

    print(splited_int)

    if not bandwidth: # check if bandwidth array is empty
        #bandwidth = splited_int
        for x in range(0, len(splited_int)):
            newset = []
            newset.append(splited_int[x])
            bandwidth.append(newset)
    else:
        for x in range(0, len(bandwidth)):
            bandwidth[x].append(splited_int[x])


#QtGui.QApplication.setGraphicsSystem('raster')
app = QtGui.QApplication([])
#mw = QtGui.QMainWindow()
#mw.resize(800,800)

win = pg.GraphicsWindow(title="Clients bandwidth monitoring")
win.resize(1500,500)
#win.setWindowTitle('Clients bandwidth monitoring')

# Enable antialiasing for prettier plots
pg.setConfigOptions(antialias=True)
label = pg.LabelItem(justify='right')
win.addItem(label)

p2 = win.addPlot(row=0, col=0, title="HLS player streaming bandwidth to MEC caching server (172.18.3.227), at last " +str(max_history_second) + " seconds")
p2.addLegend()
p2.setXRange(0,30)
p2.setYRange(0,5000)
p2.setLabel('bottom', 'Time', units='second')
p2.setLabel('left', 'Speed', units='kbps')
#p2.plot(np.random.normal(size=100), pen=(255,0,0), name="Red curve")
#p2.plot(np.random.normal(size=110)+5, pen=(0,255,0), name="Green curve")
#p2.plot(np.random.normal(size=120)+10, pen=(0,0,255), name="Blue curve")



def updateplot():
    global curve, bandwidth, label
    realtime_stats="<b>Clients' speed"
    if curve:
        for z in range(0, len(bandwidth)):
            curve[z].setData(bandwidth[z][-max_history_second:])
            #curve[z].setData(name= "Client" + str(z) + ": " + str(bandwidth[z][-1]))

            realtime_stats += "<br><span style='color: " + pg.intColor(z,6,maxValue=128).name() + "'>Client" + str(z) + ":</span> " + str(bandwidth[z][-1]) + "kbps</br>"

        realtime_stats += "</b>"
        label.setText(realtime_stats)
            
    else:
        for y in range(0, len(bandwidth)):
            curve.append(p2.plot(pen=pg.mkPen(pg.intColor(y,6,maxValue=128), width=3), name="Client"+str(y)))

    AddNewBandwidth()

timer = QtCore.QTimer()
timer.timeout.connect(updateplot)
timer.start(update_freq)


## Start Qt event loop unless running in interactive mode or using pyside.
if __name__ == '__main__':
    import sys
    AddNewBandwidth()
    time.sleep(2) # make sure the first batch of bandwidth is got
    if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
        QtGui.QApplication.instance().exec_()