FROM openmmlab/mmdeploy:ubuntu20.04-cuda11.8-mmdeploy 
RUN apt update
RUN wget -O opencv-4.9.0.zip https://codeload.github.com/opencv/opencv/zip/refs/tags/4.9.0
RUN wget -O opencv_contrib-4.9.0.zip https://codeload.github.com/opencv/opencv_contrib/zip/refs/tags/4.9.0
RUN unzip opencv-4.9.0.zip 
RUN unzip opencv_contrib-4.9.0.zip 
RUN cd opencv-4.9.0
RUN mkdir build
RUN cd build
RUN cmake -D CMAKE_BUILD_TYPE=RELEASE -D CMAKE_INSTALL_PREFIX=/usr/local -D CMAKE_C_COMPILER=/usr/bin/gcc-9 -D OPENCV_ENABLE_NONFREE=OFF -D WITH_CUDA=ON -D WITH_CUDNN=ON -D WITH_TBB=ON -D OPENCV_DNN_CUDA=ON -D ENABLE_FAST_MATH=1 -D CUDA_FAST_MATH=1 -D CUDA_ARCH_BIN=8.9 -D CUDA_ARCH_PTX="" -D WITH_CUBLAS=1 -D OPENCV_GENERATE_PKGCONFIG=ON -D OPENCV_EXTRA_MODULES_PATH=~/workspace/opencv_contrib-4.9.0/modules -D PYTHON3_EXECUTABLE=/usr/bin/python3.8 -D PYTHON3_INCLUDE_DIR=/usr/include/python3.8 -D PYTHON3_LIBRARY=/usr/lib/python3.8/config-3.8-x86_64-linux-gnu/libpython3.8.so -D PYTHON3_NUMPY_INCLUDE_DIRS=/usr/local/lib/python3.8/dist-packages/numpy/core/include -D PYTHON3_PACKAGES_PATH=/usr/local/lib/python3.8/dist-packages -D PYTHON_DEFAULT_EXECUTABLE=/usr/bin/python3.8  -D OPENCV_PYTHON3_INSTALL_PATH=/usr/local/lib/python3.8/dist-packages -D WITH_WEBP=OFF -D WITH_OPENCL=OFF -D ETHASHLCL=OFF -D ENABLE_CXX11=ON -D BUILD_EXAMPLES=OFF -D WITH_OPENGL=ON -D WITH_GSTREAMER=ON -D BUILD_OPENJPEG=ON -D WITH_V4L=ON -D WITH_QT=OFF -D BUILD_opencv_python3=ON -D BUILD_opencv_python2=OFF -D HAVE_opencv_python3=ON ~/workspace/opencv-4.9.0
RUN make -j8
RUN make install
RUN rm -r ~/workspace/opencv-4.9.0.zip opencv_contrib-4.9.0.zip opencv-4.9.0 opencv_contrib-4.9.0
RUN cd ~/workspace
RUN wget https://github.com/microsoft/onnxruntime/releases/download/v1.18.1/onnxruntime-linux-x64-gpu-1.18.1.tgz
RUN tar -zxvf onnxruntime-linux-x64-gpu-1.18.1.tgz
RUN rm onnxruntime-linux-x64-gpu-1.18.1.tgz
RUN wget https://download.pytorch.org/libtorch/cu118/libtorch-cxx11-abi-shared-with-deps-2.0.0%2Bcu118.zip
RUN unzip libtorch-cxx11-abi-shared-with-deps-2.0.0+cu118.zip
RUN rm libtorch-cxx11-abi-shared-with-deps-2.0.0+cu118.zip