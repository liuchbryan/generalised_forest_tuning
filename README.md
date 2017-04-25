# generalised_forest_tuning
Generalising Random Forest Parameter Optimisation to Include Stability and Cost



## Setup
We provide two scripts to set up the project environment and load the public datasets. 

### OS utilities/ Anaconda
The scripts assumes the following utilities exists:

- gcc
- unzip
- anaconda

**MacOSX** users should run the following if the above is not available in their machine:

```
brew install gcc
brew install unzip
curl https://repo.continuum.io/archive/Anaconda2-4.3.1-MacOSX-x86_64.sh > ./local_resources/anaconda2_install.sh
bash ./local_resources/anaconda2_install.sh -b
rm -f ./local_resources/anaconda2_install.sh
```

**Linux** users should run the following:

```
sudo apt-get install gcc
sudo apt-get install unzip
curl https://repo.continuum.io/archive/Anaconda2-4.3.1-Linux-x86_64.sh > ./local_resources/anaconda2_install.sh
bash ./local_resources/anaconda2_install.sh -b
rm -f ./local_resources/anaconda2_install.sh
```

### Project Environment Setup

The following script creates a separate environment `gft_env` in anaconda, and installs Python 2.7 and the necessary packages to run the experiment code.

```
./setup_environment.sh
```

The script requires ~500MB of network traffic and ~2G disk space.

### Loading Public Datasets
The following scipt loads the public datasets from the internet. It assumes `curl`, `unzip`, and `tar` is installed in the machine already.

```
./load_data.sh
```

It loads five files into the `local_resources` directory:

- orange_small_train.data (The features for the Orange small dataset)
- orange_small_train_appetency.labels (The appetency labels for the Orange small dataset)
- orange_small_train_churn.labels (The churn labels for the Orange small dataset)
- orange_small_train_upselling.labels (The upselling labels for the Orange small dataset)
- criteo_train.txt (The features and labels for the Criteo dataset)

The script requires ~4.5G of network traffic and ~11-12G of disk space in addition to the setup script above.

