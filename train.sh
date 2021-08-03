#!/bin/bash

trap "exit" INT
#python train.py task=segmentation models=segmentation/eye3d_kpconv model_name=KPConvPaper data=segmentation/openeds2021
python3 train.py task=segmentation models=segmentation/eye3d_kpconv model_name=KPDeformableConvPaper data=segmentation/openeds2021
#python train.py task=segmentation models=segmentation/eye3d_minkowski model_name=MinkUNet14A data=segmentation/openeds2021