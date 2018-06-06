1. To use chrome broswer to visit the web demo, please start chrome by the flag

--disable-web-security --user-data-dir --allow-file-access-from-files

2. Since the https certificate is self-signed and regard as dangerous by chrome,

Please visit port 9000 first, accept it, and then go to 8000. 
Otherwise the web demo will keep showing "Servers: Disconnected"

3. Please make sure the gpu resource is not occupied.
Run the cuda-vector-add.yaml for testing