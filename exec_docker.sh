#! /bin/bash

sudo docker run -it --name nia-test -v /home/ubuntu/Dacon/cpt_data/landmark-final:/NIA-Docker/dataset --gpus all nia-landmark
 
