FROM nvcr.io/nvidia/cuda:11.1.1-devel-ubuntu20.04
ENV TZ=Asia/Seoul
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone
ENV TORCH_CUDA_ARCH_LIST="Kepler;Kepler+Tesla;Maxwell;Maxwell+Tegra;Pascal;Volta;Turing;Ampere"
ENV FORCE_CUDA=1
ENV CUDA_HOME=/usr/local/cuda
RUN apt-get update && apt-get install -qq -y build-essential python3-dev python3-pip libopenblas-dev xvfb libgl1-mesa-glx git cmake tzdata \
            && apt-get autoremove -y && \
            apt-get clean && \
             rm -rf /var/lib/apt/lists/*
RUN pip3 install torch==1.9.0+cu111 torchvision==0.10.0+cu111 -f https://download.pytorch.org/whl/torch_stable.html --no-cache-dir
RUN pip3 install torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric -f https://pytorch-geometric.com/whl/torch-1.9.0+cu111.html
RUN pip3 install ninja h5py openmesh --no-cache-dir
RUN pip3 install -U MinkowskiEngine==0.5.4 --install-option="--blas=openblas" --install-option="--force_cuda" -v --no-deps --no-cache-dir
RUN pip3 install torch-points-kernels --no-cache-dir
RUN pip3 install git+https://github.com/nicolas-chaulet/torch-points3d.git@5378a2a --no-cache-dir