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
$ sudo apt-get install python-qt4 python-qt4-* qt4-designer python-pyqt5 python-pyqt5.qtmultimedia python3-pyqt5

Using Pyqt.Phonon, need to install gStreamer and its plugin
$sudo apt-get install libgstreamer1.0-0 gstreamer1.0-plugins-base gstreamer1.0-plugins-good gstreamer1.0-plugins-bad gstreamer1.0-plugins-ugly gstreamer1.0-libav gstreamer1.0-doc gstreamer1.0-tools
$sudo apt-get install python3-pyqt4.phonon

Install mpv player and libmpv.so
$sudo apt-get install mpv libmpv-dev

Use goaccess to analyze and plot statistic of nginx server access log
Installation: https://blog.gtwang.org/linux/analysing-nginx-logs-using-goaccess/

Command line monitoring
$ goaccess -f /var/log/nginx/access.log

HTML monitoring
$ goaccess -f /var/log/nginx/access.log -o /home/hmcheng/nginx/html/report.html --log-format=COMBINED --real-time-html --html-prefs='{"theme":"bright","perPage":10}' 

within 1 hour
$ sed -n '/'$(date '+%d\/%b\/%Y' -d '1 hour ago')'/,$ p' /var/log/nginx/access.log | goaccess -a -o /home/hmcheng/nginx/html/report.html --log-format=COMBINED --real-time-html --html-prefs='{"theme":"bright","perPage":10}' 

and go to localhost/report.html


Install nginx amplify
Additional nginx metrix

log_format  main_ext  '$remote_addr - $remote_user [$time_local] "$request" '
                      '$status $body_bytes_sent "$http_referer" '
                      '"$http_user_agent" "$http_x_forwarded_for" '
                      '"$host" sn="$server_name" '
                      'rt=$request_time '
                      'ua="$upstream_addr" us="$upstream_status" '
                      'ut="$upstream_response_time" ul="$upstream_response_length" '
                      'cs=$upstream_cache_status' ;
                      
access_log  /var/log/nginx/access.log  main_ext;
error_log  /var/log/nginx/error.log warn;


Command to get list of unique IP visiting the server
$ netstat -ntu | awk '{print $5}' | cut -d: -f1 | sort | uniq -c | sort -n

Get the most recent 20 bytes sent of the HTTP request to the server
$ sudo tail -20 /var/log/nginx/access.log| awk '{print $10}'
The corresponding remote address (i.e. client address)
$ sudo tail -20 /var/log/nginx/access.log| awk '{print $1}'

nginx access.log request time definition
$request_time
request processing time in seconds with a milliseconds resolution; time elapsed between the first bytes were read from the client and the log write after the last bytes were sent to the client


To stream rtsp to mjpeg using ffserver and ffmpeg

sudo vim /etc/ffserver.conf
MaxBandWidth 1000000

<Feed monitoring1.ffm>
File /tmp/monitoring1.ffm
FileMaxSize 50M
ACL allow 127.0.0.1
</Feed>

<Stream monitoring1.mjpg>
Feed monitoring1.ffm
Format mpjpeg
VideoCodec mjpeg
VideoQMin 1
VideoQMax 5
VideoBitRate 80000
VideoFrameRate 10
VideoBufferSize 150000
VideoSize 1280x720
NoAudio
</Stream>

$ ffserver
$ ffmpeg -i rtsp://admin:h0940232@172.18.9.100/Streaming/Channels/1 http://localhost:8090/monitoring1.ffm

Also need to open the port 8090 in centos 7
$ sudo iptables -I INPUT -p tcp -m tcp --dport 8090 -j ACCEPT
$ sudo iptables -I INPUT -p udp -m udp --dport 8090 -j ACCEPT
$ sudo service iptables save'

change upstream ip addr in nginx.conf
$sed -i -- "s/old_address/new_address/g" /etc/nginx/nginx.conf

make nginx start as machine startup
add "sudo nginx" in /etc/rc.local

change dhcp to static ip
edit /etc/network/interfaces

auto eno1
iface eno1 inet static
address 172.18.3.225
netmask 255.255.0.0
network 172.18.0.0
broadcast 172.18.255.255
gateway 172.18.0.1
dns-nameservers 8.8.8.8

