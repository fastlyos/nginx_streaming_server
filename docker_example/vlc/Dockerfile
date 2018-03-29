FROM buildpack-deps:xenial-curl

# procps is very common in build systems, and is a reasonably small package
RUN  apt-get update ; apt-get install software-properties-common -y ; add-apt-repository ppa:videolan/stable-daily ; apt-get update ; apt-get upgrade -y ; apt-get install -y --no-install-recommends \
		vlc \
	; rm -rf /var/lib/apt/lists/* ; vlc --version ; sed -i 's/geteuid/getppid/' /usr/bin/vlc