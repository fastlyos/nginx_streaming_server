#script to get latest bandwidth from the log file
for i in $(sudo find /var/lib/docker/containers -name *-json.log); do 
    count=$(sudo cat $i|grep $1|wc -l)
    if [ $count -gt 0 ];then
        result=$(sudo tail -500 $i|grep bandwidth|grep bits|tail -1|cut -d" " -f 9)
        if [ "$result" -eq "$result" ];then # it is a test of whether the string is an integer
            echo $result
            break
        fi
    fi
    
done