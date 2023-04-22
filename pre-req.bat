@echo off

Rem run this file to install all the pre-requisites but first install conda and python using their respective installers


conda install -c conda-forge cudatoolkit cudnn

pip install tensorflow-gpu

pip install scikit-learn

pip install matplotlib

pip install opencv-python