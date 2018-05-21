echo It is a script used to change the upstream addr of nginx when deployed in openshift
echo Change address from $1 to $2

sudo sed -i -- "s/$1/$2/g" /etc/nginx/nginx.conf
sudo nginx -s reload