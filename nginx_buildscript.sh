Can get previous one from
$ sudo nginx -V


$ ./auto/configure --prefix=/usr/share/nginx --sbin-path=/usr/sbin/nginx --modules-path=/usr/lib/nginx/modules --conf-path=/etc/nginx/nginx.conf --error-log-path=/var/log/nginx/error.log --http-log-path=/var/log/nginx/access.log --pid-path=/run/nginx.pid --lock-path=/var/lock/nginx.lock --user=www-data --group=www-data --build=Ubuntu --with-http_ssl_module --with-stream --with-mail=dynamic --with-http_mp4_module --with-http_flv_module --with-http_stub_status_module --add-module=/home/hmcheng/nginx/nginx-rtmp-module-master --add-module=/home/hmcheng/nginx/nginx-module-vts-master

make -j 4
sudo make install

