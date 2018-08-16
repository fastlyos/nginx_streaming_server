FROM ubuntu:16.04

USER root
LABEL maintainer="NGINX Docker Maintainers <docker-maint@nginx.com>"

# Download certificate and key from the customer portal (https://cs.nginx.com)
# and copy to the build context
COPY nginx-repo.crt /etc/ssl/nginx/
COPY nginx-repo.key /etc/ssl/nginx/

# Install NGINX Plus
RUN set -x \
  && apt-get update && apt-get upgrade -y \
  && apt-get install --no-install-recommends --no-install-suggests -y apt-transport-https lsb-release ca-certificates \
  && \
  NGINX_GPGKEY=573BFD6B3D8FBC641079A6ABABF5BD827BD9BF62; \
  found=''; \
  for server in \
    ha.pool.sks-keyservers.net \
    hkp://keyserver.ubuntu.com:80 \
    hkp://p80.pool.sks-keyservers.net:80 \
    pgp.mit.edu \
  ; do \
    echo "Fetching GPG key $NGINX_GPGKEY from $server"; \
    apt-key adv --keyserver "$server" --keyserver-options timeout=10 --recv-keys "$NGINX_GPGKEY" && found=yes && break; \
  done; \
  test -z "$found" && echo >&2 "error: failed to fetch GPG key $NGINX_GPGKEY" && exit 1; \
  echo "Acquire::https::plus-pkgs.nginx.com::Verify-Peer \"true\";" >> /etc/apt/apt.conf.d/90nginx \
  && echo "Acquire::https::plus-pkgs.nginx.com::Verify-Host \"true\";" >> /etc/apt/apt.conf.d/90nginx \
  && echo "Acquire::https::plus-pkgs.nginx.com::SslCert     \"/etc/ssl/nginx/nginx-repo.crt\";" >> /etc/apt/apt.conf.d/90nginx \
  && echo "Acquire::https::plus-pkgs.nginx.com::SslKey      \"/etc/ssl/nginx/nginx-repo.key\";" >> /etc/apt/apt.conf.d/90nginx \
  && printf "deb https://plus-pkgs.nginx.com/ubuntu `lsb_release -cs` nginx-plus\n" > /etc/apt/sources.list.d/nginx-plus.list \
  && apt-get update && apt-get install -y nginx-plus \
  #&& apt-get remove --purge --auto-remove -y gnupg1 \
  && rm -rf /var/lib/apt/lists/*
  
# Forward request logs to Docker log collector
RUN ln -sf /dev/stdout /var/log/nginx/access.log \
  && ln -sf /dev/stderr /var/log/nginx/error.log

EXPOSE 80

STOPSIGNAL SIGTERM

#copy those content to the docker image

# folder for the web plot demo
COPY demo /usr/share/nginx/html/demo
ADD index.html /usr/share/nginx/html

# folder for the nginx.conf
COPY conf /etc/nginx
COPY tools /etc/nginx

# folder for the reverse proxy cache
COPY cache /usr/share/nginx/html/cache 

RUN chmod -R 777 /var/log/nginx \
&& chmod -R 777 /var/run \
&& chmod -R 777 /etc/nginx/*
#VOLUME /usr/share/nginx/html
#VOLUME /etc/nginx

# Need to do this for file permission problem of the underlying file system
RUN chmod 777 /usr/share/nginx/html/demo/get_client_bw_loop.sh

# since dockerfile doesn't accept 2 CMD, need to run together
# not using nginx -g 'daemon off;' since want to run nginx in background and run the bandwidth 
# estimation script in foreground
CMD ["sh","-c","nginx ; /usr/share/nginx/html/demo/get_client_bw_loop.sh"]
