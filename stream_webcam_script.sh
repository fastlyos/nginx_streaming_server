ffmpeg -re -f video4linux2 -i /dev/video0 -c:v libx264 -c:a aac strict -2 -f flv rtmp://localhost/live/webcam
