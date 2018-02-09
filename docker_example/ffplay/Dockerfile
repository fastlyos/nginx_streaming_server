FROM buildpack-deps:xenial-curl

# procps is very common in build systems, and is a reasonably small package
RUN  apt-get update && apt-get install software-properties-common -y && add-apt-repository ppa:mc3man/trusty-media &&  apt-get install -y --no-install-recommends \
		ffmpeg \
	&& rm -rf /var/lib/apt/lists/* && curl 172.18.3.20 && ffplay -nodisp http://172.18.3.20/hls/sample10.m3u8
