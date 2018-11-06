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
    tar xf dlib.tar.bz2

RUN cd ~/dlib-tmp/dlib-19.13/python_examples && \
    mkdir build && \
    cd build && \
    cmake ../../tools/python -DUSE_AVX_INSTRUCTIONS=ON  -DDLIB_USE_CUDA=1&& \
    cmake --build . --config Release && \
    cp dlib.so /usr/local/lib/python2.7/dist-packages

#correct the system for logging comparison
#RUN apt-get update && apt-get install -y systemd-services
#RUN ntpdate ntp.ubuntu.com
#RUN timedatectl set-timezone HKT
#RUN ln -sf /usr/share/zoneinfo/HKT /etc/localtime



#ADD cuda_samples /root/openface/cuda_samples
#RUN cd ~/openface/cuda_samples/NVIDIA_CUDA-9.2_Samples && make


RUN apt-get install -y libjpeg-dev

#install latest ffmpegg
#RUN add-apt-repository ppa:jonathonf/ffmpeg-3 
RUN apt update
RUN apt upgrade -y
#RUN apt install -y ffmpeg 
# RUN apt install -y libav-tools
# RUN apt install -y x264
# RUN apt install -y x265
# RUN apt install -y libjpeg-turbo8-dev
# RUN ffmpeg -version

#INstall opencv with cuda enabled
RUN cd ~ && \
    mkdir -p ocv-tmp &&  \
   cd ocv-tmp &&  \
   curl -L \
	https://github.com/Itseez/opencv/archive/2.4.13.6.zip \
	-o ocv.zip &&  \
   unzip ocv.zip

#install libjpeg-turbo
RUN wget https://launchpad.net/ubuntu/+archive/primary/+files/libjpeg-turbo_1.5.1.orig.tar.gz
RUN tar xvf libjpeg-turbo_1.5.1.orig.tar.gz
RUN apt-get install -y nasm 
RUN apt-get install build-essential
RUN apt-get install -y dh-autoreconf

RUN cd libjpeg-turbo-1.5.1 && \
    autoreconf -fiv && \
    mkdir buildlibjpeg && \
    cd buildlibjpeg && \
    sh ../configure --prefix=/usr/libjpeg-turbo --mandir=/usr/share/man --with-jpeg8 --enable-static --docdir=/usr/share/doc/libjpeg-turbo-1.5.1

# 修改Makefile，把CC: gcc改成CC: gcc -fPIC
# "-fPIC" which was an abbreviation for Position Independent Code, and this had to be passed to create library code objects, without that flag, code that is specific to the source would be used, and then the library would fail.
RUN cd libjpeg-turbo-1.5.1/buildlibjpeg && \
     sed -i -- "s/CC = gcc/CC = gcc -fPIC/g" Makefile  &&\
     make && \
     make test && \
     make install

RUN add-apt-repository ppa:jonathonf/ffmpeg-3 
RUN apt-get update
RUN apt-get dist-upgrade -y
RUN apt-get install -y ffmpeg

RUN cd ~/ocv-tmp/opencv-2.4.13.6 &&  \
   mkdir release &&  \
   cd release &&  \
   cmake -D CMAKE_BUILD_TYPE=RELEASE    \
       -D CMAKE_INSTALL_PREFIX=/usr/local      \
       #-D WITH_OPENMP=ON \
     -D WITH_CUDA=ON       \
    #-D ENABLE_FAST_MATH=1     \
      -D CUDA_FAST_MATH=1      \
      -D WITH_OPENCL=OFF \
     -D WITH_CUBLAS=1      \
     -D BUILD_PYTHON_SUPPORT=ON   \
      -D WITH_FFMPEG=1 \
      -D WITH_JPEG=ON \
      -D BUILD_JPEG=OFF \
      -DJPEG_INCLUDE_DIR=/usr/libjpeg-turbo/include -DJPEG_LIBRARY=/usr/libjpeg-turbo/lib/libjpeg.a \
      #-D WITH_JPEG=1 \
     # -D OPENCV_BUILD_3RDPARTY_LIBS=1 \
     #-D CUDA_GENERATION=Auto \
     -DCUDA_ARCH_BIN="6.0" \
     -DCUDA_ARCH_PTX="6.0" \
        .. &&  \
   make -j &&  \
   make install

ADD cuda_samples /root/openface/cuda_samples
RUN chmod -R 777 /root/openface/cuda_samples/*
ADD data /root/openface/data
RUN chmod -R 777 /root/openface/data/*

#Build opencv gpu samples for testing
#RUN cd ~/ocv-tmp/opencv-2.4.13.6/samples/gpu && mkdir -p build && cd build && cmake .. && make

# a python wrapper for opencv cuda haar cascade face detector
#RUN apt-get install -y libboost-python-dev
#RUN pip install numpy
RUN pkg-config --libs python2
RUN pkg-config --cflags python2
RUN pkg-config --libs opencv
RUN pkg-config --cflags opencv
RUN ldconfig -p | grep libboost_python
RUN ls -l /usr/lib/x86_64-linux-gnu/libboost_python*

#RUN ./root/openface/cuda_samples/vectorAdd
#RUN cd /root/openface/opencv-gpu-py/test && python test.py

#Correct the timezone to GMT+8 
RUN ln -sf /usr/share/zoneinfo/Hongkong /etc/localtime
RUN apt-get install -y htop
RUN locale-gen "en_US.UTF-8"
RUN dpkg-reconfigure locales
RUN apt-get install -y locate 

RUN apt-get install -y xdot
RUN pip install imgaug
RUN pip install pympler
RUN pip install mem_top
RUN pip install objgraph


ADD opencv-gpu-py /root/openface/opencv-gpu-py
RUN chmod -R 777 /root/openface/opencv-gpu-py/*
#RUN cd /root/openface/opencv-gpu-py/src && make && ls -l
#RUN cd /root/openface/opencv-gpu-py/src && make test
#--user makes pip install packages in your home directory instead, which doesn't require any special privileges.
#[-e] <local project path>
RUN nvcc --version
#RUN cat /proc/driver/nvidia/version 
RUN cd /root/openface && pip install --user -e opencv-gpu-py

#Copy the new file that use MJPEG instead of webcam
ADD new_file/index.html /root/openface/demos/web
ADD new_file/openface-demo.js /root/openface/demos/web/js
ADD new_file/start-servers-gpu.sh /root/openface/demos/web
ADD new_file/start-servers-install.sh /root/openface/demos/web
ADD new_file/websocket-server.py /root/openface/demos/web
ADD new_file/start-servers-verbose.sh /root/openface/demos/web
ADD new_file/align_dlib.py /usr/local/lib/python2.7/dist-packages/openface/
ADD new_file/openface_server.lua /root/openface/openface
ADD new_file/main.css /root/openface/demos/web/css
ADD new_file/bootstrap.min.css /root/openface/demos/web/vendor/css
#ADD new_file/jszip.min.js /root/openface/demos/web
ADD new_file/jszip.js /root/openface/demos/web
ADD new_file/FileSaver.min.js /root/openface/demos/web
#ADD new_file/FileSaver.js /root/openface/demos/web
ADD models/dlib/shape_predictor_5_face_landmarks.dat /root/openface/models/dlib

RUN chmod -R 777 /root/openface/demos/web/*

CMD /bin/bash -l -c '/root/openface/demos/web/start-servers-gpu.sh'
#CMD /bin/bash -l -c '/root/openface/demos/web/start-servers-verbose.sh'
#CMD /bin/bash -l -c '/root/openface/demos/web/start-servers-install.sh'
