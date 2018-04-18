# 1st arg: ip address of the server providing the hls
# 2nd arg: high/low

total=0
for i in $(sudo find /var/lib/docker/containers -name *-json.log); do 
    total=$(( $total + $(sudo cat $i|grep $1|grep location=|grep $2|wc -l)))
done
echo $total