#!/bin/bash

# Setup script for MacOSX (anaconda only)

# Download and install anaconda
curl https://repo.continuum.io/archive/Anaconda2-4.3.1-MacOSX-x86_64.sh > ./local_resources/anaconda2_install.sh
bash ./local_resources/anaconda2_install.sh -b

rm -f ./local_resources/anaconda2_install.sh

# Create new environment to house all libraries the project needs
conda create --yes --name gft_env python=2.7

source activate gft_env

# Install python libraries
conda install --yes numpy libgfortran pandas scikit-learn jupyter

# Using pip to install matplotlib to avoid error
pip install matplotlib

# Install Pybo
pip install -r https://github.com/mwhoffman/pybo/raw/master/requirements.txt
pip install git+https://github.com/mwhoffman/pybo.git

source deactivate

