echo This is a script written to switch between low and high upload bandwidth of the stream host to faciliate debugging hls client ability to adaptive bitrate smoothly
echo Switch between every $1 seconds

while true
do
	sudo wondershaper -a enp0s25 -c; sleep 0.2s; sudo wondershaper -a enp0s25 -u 300;
	echo Switch to low now
	sleep $1
	sudo wondershaper -a enp0s25 -c; sleep 0.2s; sudo wondershaper -a enp0s25 -u 1500;
	echo Switch to high now
	sleep $1
done	
