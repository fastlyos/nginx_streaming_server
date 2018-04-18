echo script to do counting on the high/low bitrate streamed in the containers...

echo print all logging to process...
sudo find /var/lib/docker -name *-json.log

printf "\nhigh, low\n"
for i in $(sudo find /var/lib/docker -name *-json.log); do 
    high=$(sudo cat $i|grep 172.18.3|grep location=|grep high|wc -l)
    low=$(sudo cat $i|grep 172.18.3|grep location=|grep low|wc -l)
    #echo $i, high: $high, low: $low
    echo $high, $low
done
