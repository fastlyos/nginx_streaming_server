 
`sudo rpm -i cuda-repo-rhel7-9-2-local-9.2.88-1.x86_64.rpm`
`sudo yum clean all`
`sudo yum install cuda`


 cuda-install-samples-9.2.sh cuda_examples
 
 export PATH=/usr/local/cuda-9.2/bin${PATH:+:${PATH}}
 export LD_LIBRARY_PATH=/usr/local/cuda-9.2/lib\
                         ${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}