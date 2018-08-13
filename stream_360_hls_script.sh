ffmpeg -i http://172.18.4.25:8000/sdcard/liveStream/live.m3u8  -c:v libx264 -f flv rtmp://localhost/live/360
