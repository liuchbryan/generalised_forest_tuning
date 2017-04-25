# generalised_forest_tuning
Generalising Random Forest Parameter Optimisation to Include Stability and Cost



## Setup
We provide two scripts to set up the project environment and load the public datasets. 

### Project Environment Setup 
The following **linux** script install the libraries and packages required for the project:

- gcc
- unzip (for unzip-ing a data file from the public datasets)
- anaconda (with a separate environment `gft_env` for Python 2.7, numpy, pandas, scikit-learn, jupyter, matplotlib, and pybo)

```
./setup_environment_linux.sh
```

The script requires ~500MB of network traffic and ~2G disk space.

------

For **MacOSX** users, you can install anaconda and setup the project environment with the following script:

```
./setup_environment_macosx.sh
```

You will have to ensure `gcc` and `unzip` is installed before proceeding to the next step (we recommend using `brew`). 


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

