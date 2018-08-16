#building 
$ sudo docker build --no-cache -t nginxplus .

#run, since host may be using port 80 already, use 8081 instead
$ sudo docker run --name mynginxplus -p 8081:80 -d nginxplus

# see the port mapping
$ sudo docker ps

#login to docker hub
$ sudo docker login
$ sudo docker tag nginxplus mandog/nginx_cache_server_web_demo
$ sudo docker push mandog/nginx_cache_server_web_demo

#Run a shell to access the volume (i.e. running a helper container)
$ sudo docker run -it --volumes-from mynginxplus --name mynginx_files debian /bin/bash

