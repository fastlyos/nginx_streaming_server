ffmpeg -re -i ~/video_sample/sample10.mp4 -c:a aac -strict -2 -c:v libx264 -f flv rtmp://localhost/live/zzz
