tail -20 /etc/nginx/logs/access.log | grep hls. | awk 'BEGIN{ PROCINFO["sorted_in"]="@val_num_desc" }
     { if( $11 > 0 ){ a[$1]++; b[$1]+=(0.5*(($10/($11))/1024.0)); d[$1]=$13} }
     END{ 
         print "Client_ip, bandwidth(kbps)"
         for(i in a) { if(++c>10) break; print i, ",", b[i]/a[i], ",", d[i] } 
     }' 
