tail -50 /etc/nginx/logs/access.log | grep hls. | sed 's/:/ /g' |awk 'BEGIN{ PROCINFO["sorted_in"]="@val_num_desc" }
     {
	 a[$1]++; 

     if (a[$1] < 15)
     {
        b[$1]+= $13; 
        d[$1]=$16; 
        e[$1]=$10;

        if(a[$1] == 1){
            start_sec[$1]=$5*3600 + $6*60 + $7;
        }
        end_sec[$1] =$5*3600 + $6*60 + $7;
     }

	}
     END{ 
         print "Client_ip, bandwidth(kBps), url"
         for(i in a)
	 { 
		if(++c>10) break; 
        time_interval = end_sec[i] - start_sec[i] + 1.0
        if(time_interval > 1)
        {
            print i, ",", (0.7*(b[i])/time_interval)/1024.0, ",", d[i], ",", e[i] , ",", time_interval
        }
		
	 } 
     }' 
