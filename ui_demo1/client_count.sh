# 1st arg: ip address of the server providing the hls

total=0
for i in $(sudo find /var/lib/docker/containers -name *-json.log); do 
    count=$(sudo cat $i|grep $1|wc -l)
    if [ $count -gt 0 ];then
        total=$((total+1))
    fi
done
echo $total