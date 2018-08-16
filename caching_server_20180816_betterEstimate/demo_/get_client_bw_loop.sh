#!/bin/sh
echo A script to get client bandwidth in infinite loop and write the output to bandwidth.txt

while true
do
	now=$(date +"%T")
	echo "Current time : $now"
	sh /usr/share/nginx/html/demo/get_client_bw.sh | tee /usr/share/nginx/html/demo/csv/examples/bandwidth.txt
	printf "\n"
	sleep 1 # update every one second
done

