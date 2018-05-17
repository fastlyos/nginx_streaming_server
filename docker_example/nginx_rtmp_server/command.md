#build the image
$ sudo docker build -t nginxmnt .

#run, since host may be using port 80 already, use 8082 instead
$ sudo docker run --name mynginxmnt -p 8082:80 -p 8083:1935 -d nginxmnt

# see the port mapping
$ sudo docker ps

#login to docker hub
$ sudo docker login
$ sudo docker tag nginxmnt mandog/nginx_mnt_rtmp
$ sudo docker push mandog/nginx_mnt_rtmp

#Run a shell to access the volume (i.e. running a helper container)
$ sudo docker run -it --volumes-from mynginxmnt --name mynginxmnt_files debian /bin/bash