FROM buildpack-deps:xenial-curl

# procps is very common in build systems, and is a reasonably small package
RUN  apt-get update ; apt-get install software-properties-common -y ; add-apt-repository ppa:videolan/stable-daily ; apt-get update ; apt-get upgrade -y ; apt-get install -y --no-install-recommends vlc; \
 apt-get install -y xauth \
	; rm -rf /var/lib/apt/lists/* ; \
  xauth add ubuntu3/unix:0  MIT-MAGIC-COOKIE-1  d96d8d62c032072f157f3e531305c876;\
  vlc --version ; sed -i 's/geteuid/getppid/' /usr/bin/vlc; 
