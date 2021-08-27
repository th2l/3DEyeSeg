
## :2nd_place_medal 3D Eye Point Cloud Segmentation Challenge - OpenEDS 2021

Information about 3D Eye Point Cloud Segmentation can be found in [here](https://research.fb.com/programs/facebook-openeds-2021-challenge/).

### How to use
1. Use conda environment
      - Create a conda environment with python 3.7
      - Activate the environment and use `install_packages.sh` to install required packages
      - Modify `conf/eval.yaml`, `checkpoint_dir` in line 9 to absolute path of ckpt (in current folder)
      - Run `python eval.py`

2. Use docker (the method that we used)
    - Create docker image with dockerfile, for example docker docker build -t prj:eyeseg3d .
    - Run
 ```
 docker run --gpus all --ipc=host -it --rm \
                    -v path_to_src:/mnt/Work/OpenEDS2021_EyeSeg3D \
                    -w /mnt/Work/OpenEDS2021_EyeSeg3D \
                    prj:eyeseg3d python3 eval.py
```

   In here `path_to_src` is the absolute path to source code folder.

The predictions for test set will be saved to `ckpt/eval/%Y-%m-%d_%H-%M-%S/viz/100/test`.

### Credits
Our source code is built upon [torch-points3d](https://github.com/nicolas-chaulet/torch-points3d) library.
