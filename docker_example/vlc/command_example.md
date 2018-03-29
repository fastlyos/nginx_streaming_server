start single instance with ffplay
sudo docker run -it vlc_img cvlc -vv -Vdummy http://172.18.3.227/hls/sample9.m3u8

build the docker file
sudo docker build -t vlc_img .

run docker-compose
sudo docker-compose up -d --scale app=3

kill (i.e. stop and remove) all containers
sudo docker kill $(sudo docker ps -a -q)
