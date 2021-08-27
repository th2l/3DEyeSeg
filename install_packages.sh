pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 -f https://download.pytorch.org/whl/torch_stable.html --no-cache-dir
pip install torch-scatter -f https://pytorch-geometric.com/whl/torch-1.9.0+cu111.html --no-cache-dir
pip install torch-sparse -f https://pytorch-geometric.com/whl/torch-1.9.0+cu111.html --no-cache-dir
pip install torch-geometric==1.7.2 --no-cache-dir
pip install torch-cluster -f https://pytorch-geometric.com/whl/torch-1.9.0+cu111.html --no-cache-dir
pip install torch-spline-conv -f https://pytorch-geometric.com/whl/torch-1.9.0+cu111.html --no-cache-dir
pip install ninja h5py  --no-cache-dir
FORCE_CUDA=1 pip install torch-points-kernels --no-cache-dir
pip install -U MinkowskiEngine --install-option="--blas=openblas" -v --no-deps --no-cache-dirpy
pip install git+https://github.com/nicolas-chaulet/torch-points3d.git@5378a2a --no-cache-dir