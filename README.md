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
-docker-compose (version 1.18.0 or above)
-vlc (version 2.2.7 or higher) ( Need some modification in order to run as root)

build image with vlc installed.
Use docker-compose the scale the number of instance
and run vlc-player without display (cvlc -Vdummy xxx.m3u8)

(Vlc player is tested to be able to adapt HLS bitrate automatically)

To monitor traffic, server can install:
- wondershaper (clone from git and sudo make install)
- nethogs 
- etherape
