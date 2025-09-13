FROM nvcr.io/nvidia/tensorrt:25.02-py3
RUN apt update
RUN wget -O opencv-4.11.0.zip https://codeload.github.com/opencv/opencv/zip/refs/tags/4.11.0
RUN wget -O opencv_contrib-4.11.0.zip https://codeload.github.com/opencv/opencv_contrib/zip/refs/tags/4.11.0
RUN unzip opencv-4.11.0.zip 
RUN unzip opencv_contrib-4.11.0.zip 
RUN cd opencv-4.11.0
RUN mkdir build
RUN cd build
RUN cmake  -D CMAKE_BUILD_TYPE=RELEASE -D CMAKE_INSTALL_PREFIX=/usr/local -D CMAKE_C_COMPILER=/usr/bin/gcc-13 -D OPENCV_ENABLE_NONFREE=OFF -D WITH_CUDA=ON -D WITH_CUDNN=ON -D WITH_TBB=ON -D OPENCV_DNN_CUDA=ON -D ENABLE_FAST_MATH=1 -D CUDA_FAST_MATH=1 -D CUDA_ARCH_PTX="" -D WITH_CUBLAS=1 -D OPENCV_GENERATE_PKGCONFIG=ON -D OPENCV_EXTRA_MODULES_PATH=/workspace/opencv_contrib-4.11.0/modules -D PYTHON3_EXECUTABLE=/usr/bin/python3.12 -D PYTHON3_INCLUDE_DIR=/usr/include/python3.12 -D PYTHON3_LIBRARY=/usr/lib/x86_64-linux-gnu/libpython3.12.so -D PYTHON3_NUMPY_INCLUDE_DIRS=/usr/local/lib/python3.12/dist-packages/numpy/core/include -D PYTHON3_PACKAGES_PATH=/usr/local/lib/python3.12/dist-packages -D PYTHON_DEFAULT_EXECUTABLE=/usr/bin/python3.12 -D OPENCV_PYTHON3_INSTALL_PATH=/usr/local/lib/python3.12/dist-packages -D WITH_WEBP=OFF -D WITH_OPENCL=OFF -D ETHASHLCL=OFF -D ENABLE_CXX11=ON -D BUILD_EXAMPLES=OFF -D WITH_OPENGL=ON -D WITH_GSTREAMER=ON -D BUILD_OPENJPEG=ON -D WITH_V4L=ON -D WITH_QT=OFF -D BUILD_opencv_python3=ON -D BUILD_opencv_python2=OFF -D HAVE_opencv_python3=ON /workspace/opencv-4.11.0
RUN make -j8
RUN make install -j8
RUN rm -r /workspace/opencv-4.11.0.zip opencv_contrib-4.11.0.zip opencv-4.11.0 opencv_contrib-4.11.0
RUN cd /workspace
RUN wget https://download.pytorch.org/libtorch/cu128/libtorch-cxx11-abi-shared-with-deps-2.7.0%2Bcu128.zip
RUN unzip libtorch-cxx11-abi-shared-with-deps-2.7.0+cu128.zip
RUN rm libtorch-cxx11-abi-shared-with-deps-2.7.0+cu128.zip
RUN wget https://github.com/microsoft/onnxruntime/releases/download/v1.22.0/onnxruntime-linux-x64-gpu-1.22.0.tgz
RUN tar -zxvf onnxruntime-linux-x64-gpu-1.22.0.tgz
RUN rm onnxruntime-linux-x64-gpu-1.22.0.tgz
RUN wget https://storage.openvinotoolkit.org/repositories/openvino/packages/2025.2/linux/openvino_toolkit_ubuntu24_2025.2.0.19140.c01cd93e24d_x86_64.tgz
RUN tar -zxvf openvino_toolkit_ubuntu24_2025.2.0.19140.c01cd93e24d_x86_64.tgz
RUN rm openvino_toolkit_ubuntu24_2025.2.0.19140.c01cd93e24d_x86_64.tgz
RUN pip install torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu128
RUN pip install onnxruntime-gpu==1.22.0 -i https://pypi.mirrors.ustc.edu.cn/simple/
RUN pip install openvino==2025.2.0 -i https://pypi.mirrors.ustc.edu.cn/simple/
RUN pip install cuda-python==12.8.0 -i https://pypi.mirrors.ustc.edu.cn/simple/