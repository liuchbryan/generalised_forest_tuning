#!/bin/bash

# Load Orange small data
curl http://www.vincentlemaire-labs.fr/kddcup2009/orange_small_train_appetency.labels > ./local_resources/orange_small_train_appetency.labels
curl http://www.vincentlemaire-labs.fr/kddcup2009/orange_small_train_churn.labels > ./local_resources/orange_small_train_churn.labels
curl http://www.vincentlemaire-labs.fr/kddcup2009/orange_small_train_upselling.labels > ./local_resources/orange_small_train_upselling.labels
curl http://www.vincentlemaire-labs.fr/kddcup2009/orange_small_train.data.zip > ./local_resources/orange_small_train.data.zip
unzip ./local_resources/orange_small_train.data.zip orange_small_train.data -d ./local_resources
rm -f ./local_resources/orange_small_train.data.zip

# Load Criteo data
curl https://s3-eu-west-1.amazonaws.com/criteo-labs/dac.tar.gz > ./local_resources/dac.tar.gz
tar -zxf ./local_resources/dac.tar.gz train.txt
mv train.txt ./local_resources/criteo_train.txt
rm -f ./local_resources/dac.tar.gz

