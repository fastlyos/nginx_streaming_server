start single instance with vlc
sudo docker run -it vlc_img cvlc -vv -Vdummy http://172.18.3.227/hls/sample9.m3u8

with vlc gui
sudo docker run -it --net=host -e DISPLAY -v /tmp/.X11-unix vlc_img
#vlc http://172.18.3.227/hls/sample9.m3u8

build the docker file
sudo docker build -t vlc_img .

run docker-compose
sudo docker-compose restart
DOCKER_CLIENT_TIMEOUT=200 COMPOSE_HTTP_TIMEOUT=200 sudo docker-compose up -d --scale app=3
app=200 is okay in current server, supermicro SYS-501D-FN4T, htop shows 80-90% cpu loading

$sudo docker-compose -f docker-compose-gui.yml up -d --scale app=2

kill (i.e. stop and remove) all containers
sudo docker kill $(sudo docker ps -a -q)

sudo docker stop $(sudo docker ps -a -q)
sudo docker rm $(sudo docker ps -a -q)

The containers logging is in:
/var/lib/docker/containers/<container id>/<container id>-json.log
Need to sudo -i 
first in order to broswe /var/lib/docker

Find all *-json.log in /var/lib/docker/containers
$ find . -name *-json.log
