sudo docker run -it --net host --cpuset-cpus 0 --memory 512mb -v /tmp/.X11-unix:/tmp/.X11-unix -e DISPLAY=unix$DISPLAY -v $HOME/Downloads:/root/Downloads --device /dev/snd --name arcwelder3 --security-opt seccomp:unconfined thshaw/arc-welder^C

