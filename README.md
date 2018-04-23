Running environment: Ubuntu 16.04 server LTS

Need to install nginx manually with the rtmp module

Additional tools:
-ffmpeg3

$ sudo add-apt-repository ppa:jonathonf/ffmpeg-3
$ sudo apt update
$ sudo apt install ffmpeg libav-tools x264 x265

-pcre
-openssl

To simulate the client traffic, 
we use
-docker
-docker-compose (version 1.20.0 or above, support variable substitution)
-vlc (version 2.2.7 or higher) ( Need some modification in order to run as root)

build image with vlc installed.
Use docker-compose the scale the number of instance
and run vlc-player without display (cvlc -Vdummy xxx.m3u8)

(Vlc player is tested to be able to adapt HLS bitrate automatically)

To monitor traffic, server can install:
- wondershaper (clone from git and sudo make install)
- nethogs 
- etherape

To test for http request timing
$ curl -s -w "%{time_total}\n" -o /dev/null http://172.18.3.225

To test the traffic in nginx, use "ngxtop"
$ pip install ngxtop
$ ngxtop -l /var/log/nginx/access.log

To run the vlc_gui demo.py, need to install pyQt4 (also qt-creator if need to edit)
$ sudo apt-get install python-qt4 qt4-designer python-pyqt5 python-pyqt5.qtmultimedia

Using Pyqt.Phonon, need to install gStreamer and its plugin
$sudo apt-get install libgstreamer1.0-0 gstreamer1.0-plugins-base gstreamer1.0-plugins-good gstreamer1.0-plugins-bad gstreamer1.0-plugins-ugly gstreamer1.0-libav gstreamer1.0-doc gstreamer1.0-tools
