docker run --gpus all --ipc=host -it --rm \
        -v /mnt/Work/OneDrive_backup_data/PRJ_src:/mnt/Work/OpenEDS2021_EyeSeg3D \
        -w /mnt/Work/OpenEDS2021_EyeSeg3D \
        prj:eyeseg3d python3 eval.py
