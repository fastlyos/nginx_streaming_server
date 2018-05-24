FROM ubuntu:16.04

USER root
#Install ffmpeg for transcoding
RUN  apt-get update && apt-get install software-properties-common build-essential -y && add-apt-repository ppa:mc3man/trusty-media &&  apt-get install -y  \
		ffmpeg libav-tools x264 x265 libpcre3 libpcre3-dev openssl libssl-dev\
	&& rm -rf /var/lib/apt/lists/* 

# install nginx with 3rd party rtmp
COPY nginx /usr/local/nginx
WORKDIR /usr/local/nginx
RUN ./auto/configure --prefix=/usr/share/nginx --sbin-path=/usr/sbin/nginx --modules-path=/usr/lib/nginx/modules --conf-path=/etc/nginx/nginx.conf --error-log-path=/var/log/nginx/error.log --http-log-path=/var/log/nginx/access.log --pid-path=/run/nginx.pid --lock-path=/var/lock/nginx.lock --user=www-data --group=www-data --build=Ubuntu --with-http_ssl_module --with-stream --with-mail=dynamic --with-http_mp4_module --with-http_flv_module --with-http_stub_status_module --add-module=nginx-rtmp-module-master --add-module=nginx-module-vts-master --add-module=echo-nginx-module
RUN make -j 4
RUN make install
RUN nginx -V


#copy those content to the docker image
COPY mnt /usr/share/nginx/html/mnt
COPY conf /etc/nginx

RUN chmod -R 777 /var/log/nginx \
&& chmod -R 777 /var/run \
&& chmod -R 777 /etc/nginx/*

#VOLUME /usr/share/nginx/html
#VOLUME /etc/nginx

EXPOSE 80
#1935 is standard rtmp port
EXPOSE 1935

CMD ["nginx", "-g", "daemon off;"]
