#!/bin/bash

# Setup script for Linux

sudo apt-get install gcc
sudo apt-get install unzip

# Download and install anaconda
curl https://repo.continuum.io/archive/Anaconda2-4.3.1-Linux-x86_64.sh > ~/anaconda2_install.sh
bash anaconda2_install.sh -b -f

rm -f anaconda2_install.sh

# Create new environment to house all libraries the project needs
conda create --yes --name gft_env python=2.7

source activate gft_env

# Install python libraries
conda install --yes numpy pandas scikit-learn jupyter

# Using pip to install matplotlib to avoid error
pip install matplotlib

# Install Pybo
pip install -r https://github.com/mwhoffman/pybo/raw/master/requirements.txt
pip install git+https://github.com/mwhoffman/pybo.git

# Update all packages
conda update --yes --all
# Force update libgfortran due to the following seemingly bug:
# https://github.com/ContinuumIO/anaconda-issues/issues/686#issuecomment-221102256
conda update --force --yes libgfortran

source deactivate

