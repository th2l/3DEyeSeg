docker run --gpus all --ipc=host -it --rm \
        --user $UID:$GID \
        --volume="/etc/group:/etc/group:ro" \
        --volume="/etc/passwd:/etc/passwd:ro" \
        --volume="/etc/shadow:/etc/shadow:ro" \
        -v /mnt/Work/OneDrive_backup_data/OpenEDS2021_EyeSeg3D:/mnt/Work/OneDrive_backup_data/OpenEDS2021_EyeSeg3D \
        -v /mnt/XProject/OpenEDS2021_EyeSeg3D:/mnt/XProject/OpenEDS2021_EyeSeg3D \
        -w /mnt/XProject/OpenEDS2021_EyeSeg3D \
        openeds2021:eyeseg3d bash