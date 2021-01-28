#! /bin/bash

sudo docker run -it --name nia -v /home/ubuntu/Dacon/HDD_02/landmark/data:/Landmark-Recognition/data --gpus all nia-landmark
 
