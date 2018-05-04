tail -1000 /etc/nginx/logs/access.log | awk 'BEGIN{ PROCINFO["sorted_in"]="@val_num_desc" }
     { if( $15 > 0 ){ a[$1]++; b[$1]+=(($10/($15))/1024)} }
     END{ 
         #print "Client ip, Total request, estimated bandwidth(kBps) per second"
         for(i in a) { if(++c>10) break; print i, b[i]/a[i] } 
     }' 
