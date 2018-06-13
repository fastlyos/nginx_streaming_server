# from ubuntu 14.04 with opencv, dlib, torch installed
#FROM bamos/openface
#FROM mandog/mjpeg_openface
FROM mandog/openface_cuda

#install cuda 8.0 runtime
# RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates apt-transport-https gnupg-curl && \
#     rm -rf /var/lib/apt/lists/* && \
#     NVIDIA_GPGKEY_SUM=d1be581509378368edeec8c1eb2958702feedf3bc3d17011adbf24efacce4ab5 && \
#     NVIDIA_GPGKEY_FPR=ae09fe4bbd223a84b2ccfce3f60f4b3d7fa2af80 && \
#     apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1404/x86_64/7fa2af80.pub && \
#     apt-key adv --export --no-emit-version -a $NVIDIA_GPGKEY_FPR | tail -n +2 > cudasign.pub && \
#     echo "$NVIDIA_GPGKEY_SUM  cudasign.pub" | sha256sum -c --strict - && rm cudasign.pub && \
#     echo "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1404/x86_64 /" > /etc/apt/sources.list.d/cuda.list

# ENV CUDA_VERSION 8.0.61

# ENV CUDA_PKG_VERSION 8-0=$CUDA_VERSION-1
# RUN apt-get update && apt-get install -y --no-install-recommends \
#         cuda-nvrtc-$CUDA_PKG_VERSION \
#         cuda-nvgraph-$CUDA_PKG_VERSION \
#         cuda-cusolver-$CUDA_PKG_VERSION \
#         cuda-cublas-8-0=8.0.61.2-1 \
#         cuda-cufft-$CUDA_PKG_VERSION \
#         cuda-curand-$CUDA_PKG_VERSION \
#         cuda-cusparse-$CUDA_PKG_VERSION \
#         cuda-npp-$CUDA_PKG_VERSION \
#         cuda-cudart-$CUDA_PKG_VERSION && \
#     ln -s cuda-8.0 /usr/local/cuda && \
#     rm -rf /var/lib/apt/lists/*

# # nvidia-docker 1.0
# LABEL com.nvidia.volumes.needed="nvidia_driver"
# LABEL com.nvidia.cuda.version="${CUDA_VERSION}"

# RUN echo "/usr/local/nvidia/lib" >> /etc/ld.so.conf.d/nvidia.conf && \
#     echo "/usr/local/nvidia/lib64" >> /etc/ld.so.conf.d/nvidia.conf

# ENV PATH /usr/local/nvidia/bin:/usr/local/cuda/bin:${PATH}
# ENV LD_LIBRARY_PATH /usr/local/nvidia/lib:/usr/local/nvidia/lib64

# # nvidia-container-runtime
# ENV NVIDIA_VISIBLE_DEVICES all
# ENV NVIDIA_DRIVER_CAPABILITIES compute,utility
# ENV NVIDIA_REQUIRE_CUDA "cuda>=8.0"



# # devel
# RUN apt-get update && apt-get install -y --no-install-recommends \
#         cuda-core-$CUDA_PKG_VERSION \
#         cuda-misc-headers-$CUDA_PKG_VERSION \
#         cuda-command-line-tools-$CUDA_PKG_VERSION \
#         cuda-nvrtc-dev-$CUDA_PKG_VERSION \
#         cuda-nvml-dev-$CUDA_PKG_VERSION \
#         cuda-nvgraph-dev-$CUDA_PKG_VERSION \
#         cuda-cusolver-dev-$CUDA_PKG_VERSION \
#         cuda-cublas-dev-8-0=8.0.61.2-1 \
#         cuda-cufft-dev-$CUDA_PKG_VERSION \
#         cuda-curand-dev-$CUDA_PKG_VERSION \
#         cuda-cusparse-dev-$CUDA_PKG_VERSION \
#         cuda-npp-dev-$CUDA_PKG_VERSION \
#         cuda-cudart-dev-$CUDA_PKG_VERSION \
#         cuda-driver-dev-$CUDA_PKG_VERSION && \
#     rm -rf /var/lib/apt/lists/*

# ENV LIBRARY_PATH /usr/local/cuda/lib64/stubs:${LIBRARY_PATH}





# # install cutorch
#RUN for NAME in torch dpnn nn optim optnet csvigo fblualib torchx tds; do luarocks install $NAME; done
RUN luarocks install torch
RUN luarocks install dpnn
RUN luarocks install nn
RUN luarocks install optim
RUN luarocks install optnet
RUN luarocks install csvigo
#RUN luarocks install fblualib
RUN luarocks install torchx
RUN luarocks install tds

RUN luarocks install FindCUDA
RUN luarocks install cutorch
RUN luarocks install cunn

#CMD /bin/bash -l -c '/root/openface/demos/web/start-servers.sh'


#Install cudnn v7 runtime for dlib to use cuda
# RUN echo "deb https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1404/x86_64 /" > /etc/apt/sources.list.d/nvidia-ml.list

# ENV CUDNN_VERSION 5.1.10
# LABEL com.nvidia.cudnn.version="${CUDNN_VERSION}"

# RUN apt-get update && apt-get install -y --no-install-recommends \
#             libcudnn5=$CUDNN_VERSION-1+cuda8.0 \
#             libcudnn5-dev=$CUDNN_VERSION-1+cuda8.0 && \
#     rm -rf /var/lib/apt/lists/*

# ADD libcudnn7_7.1.4.18-1+cuda8.0_amd64.deb .
# RUN dpkg -i libcudnn7_7.1.4.18-1+cuda8.0_amd64.deb

ADD cuda/include/cudnn.h /usr/local/cuda/include
ADD cuda/lib64/libcudnn* /usr/local/cuda/lib64/
RUN chmod a+r /usr/local/cuda/include/cudnn.h /usr/local/cuda/lib64/libcudnn*

#check if cudnn is properly installed
RUN ls /usr/local/cuda/lib64
RUN cat /usr/local/cuda/include/cudnn.h | grep CUDNN_MAJOR -A 2


#Install dlib with AVX on
#Can check AVX instruction set is supported by $grep avx /proc/cpuinfo
RUN cd ~ && \
    mkdir -p dlib-tmp &&  \
    cd dlib-tmp && \
    curl -L \
         https://github.com/davisking/dlib/archive/v19.13.tar.gz \
         -o dlib.tar.bz2 && \
    tar xf dlib.tar.bz2 && \
    cd dlib-19.13/python_examples && \
    mkdir build && \
    cd build && \
    cmake ../../tools/python -DUSE_AVX_INSTRUCTIONS=ON && \
    cmake --build . --config Release && \
    cp dlib.so /usr/local/lib/python2.7/dist-packages

#correct the system for logging comparison
#RUN apt-get update && apt-get install -y systemd-services
#RUN ntpdate ntp.ubuntu.com
#RUN timedatectl set-timezone HKT
#RUN ln -sf /usr/share/zoneinfo/HKT /etc/localtime

ADD cuda_samples2 /root/openface/cuda_samples

#ADD cuda_samples /root/openface/cuda_samples
#RUN cd ~/openface/cuda_samples/NVIDIA_CUDA-9.2_Samples && make

#Copy the new file that use MJPEG instead of webcam
ADD new_file/index.html /root/openface/demos/web
ADD new_file/openface-demo.js /root/openface/demos/web/js
ADD new_file/start-servers-gpu.sh /root/openface/demos/web
ADD new_file/start-servers-install.sh /root/openface/demos/web
ADD new_file/websocket-server.py /root/openface/demos/web
ADD new_file/start-servers-verbose.sh /root/openface/demos/web
ADD new_file/align_dlib.py /root/openface/openface
ADD new_file/main.css /root/openface/demos/web/css

RUN chmod -R 777 /root/openface/demos/web/*

CMD /bin/bash -l -c '/root/openface/demos/web/start-servers-gpu.sh'
#CMD /bin/bash -l -c '/root/openface/demos/web/start-servers-verbose.sh'
#CMD /bin/bash -l -c '/root/openface/demos/web/start-servers-install.sh'


