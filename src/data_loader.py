import pandas as pd # You need pandas 0.19+
import numpy as np
from sklearn import preprocessing
from sklearn.pipeline import Pipeline
import random


def get_and_process_orange_small_data():
    ## FEATURES
    # For the orange small dataset, the first 190 variables are numerical and
    # the last 40 are categorical. Variables come as "VarX", X from 1 to 230.

    num_cols = 190
    cat_cols = 40

    data_dtype = {
        "Var" + str(i): (np.float64 if i <= num_cols else 'category')
        for i in range(1, num_cols + cat_cols + 1)
        }

    data_num_cols = [
        "Var" + str(i) for i in range(1, num_cols + 1)
        ]

    data_cat_cols = [
        "Var" + str(i) for i in range(num_cols + 1, num_cols + cat_cols + 1)
        ]

    # These categorical variables have too many possible values (>1000),
    # not using them in this study as they hinder expressing our points
    # on parameter-tuning effectively
    data_too_many_cat_cols = [
        "Var198", "Var199", "Var200", "Var202", "Var214", "Var216",
        "Var217", "Var220", "Var222"
    ]

    data_cat_cols_used = \
        list(set(data_cat_cols) - set(data_too_many_cat_cols))

    data = pd.read_csv(
        "../local_resources/orange_small_train.data",
        sep="\t",
        dtype=data_dtype
    )

    # Pre-processing for numerical data
    data_num = data[data_num_cols]

    feature_num_imputer = preprocessing.Imputer(missing_values='NaN',
                                                strategy='most_frequent')
    feature_num_stdscaler = preprocessing.StandardScaler()

    feature_num_preproc_pipeline = \
        Pipeline([('imputer', feature_num_imputer),
                  ('stdscaler', feature_num_stdscaler)])

    feature_num_preproc_pipeline.fit(data_num)

    data_num = feature_num_preproc_pipeline.transform(data_num)

    # Pre-processing for categorical data
    data_cat = \
        pd.get_dummies(data[data_cat_cols_used], dummy_na=True)

    features = np.concatenate((data_num, data_cat), axis=1)

    ## LABELS
    # Load the labels, and process it as scikit-learn compatible output
    label_binarizer = preprocessing.Binarizer(threshold=0)

    # Appentency
    appentency_labels = \
        label_binarizer.transform(
            pd.read_csv(
                "../local_resources/orange_small_train_appetency.labels",
                dtype=np.int32, header=None, names=["label"]
            )
        ).flatten()

    # Churn
    churn_labels = \
        label_binarizer.transform(
            pd.read_csv(
                "../local_resources/orange_small_train_churn.labels",
                dtype=np.int32, header=None, names=["label"]
            )
        ).flatten()


    # Upselling
    upselling_labels = \
        label_binarizer.transform(
            pd.read_csv(
                "../local_resources/orange_small_train_upselling.labels",
                dtype=np.int32, header=None, names=["label"]
            )
        ).flatten()

    return features, appentency_labels, churn_labels, upselling_labels


def get_and_process_criteo_data(sample=True):
    num_cols = 12
    cat_cols = 25

    names = ['label']
    names = names + ['Int_'+str(x) for x in range(num_cols + 1)]
    names = names + ['Cat_'+str(x) for x in range(cat_cols + 1)]

    df = pd.read_table('../local_resources/criteo_train.txt', 
                       sep='\t', names=names)
    
    if sample:
        df = df.sample(frac=0.1, random_state=42)

    #Drop columns that have > 100 categories
    drop_list = []

    for i in range(cat_cols + 1):
        num_cats = len(df['Cat_'+str(i)].unique())
        
        if (num_cats > 100):
            drop_list.append('Cat_'+str(i))

    df = df.drop(drop_list, axis = 1)

    #Fill missing data with mode of column
    for column in df.columns:
        df[column] = df[column].fillna(df[column].mode()[0])

    #Fill 'unnamed' data with mode
    for column in df.columns:
        column_mode = df[column].mode()[0]
        df[column] = df[column].apply(lambda x : column_mode if 'Unnamed' in str(x) else x)

    #Scale numerical data
    df[df.columns[1:num_cols + 2]] = \
        preprocessing.StandardScaler().fit_transform(df[df.columns[1:num_cols + 2]])

    #Convert categorical columns into binary columns
    binary_cats = pd.get_dummies(df[df.columns[num_cols + 2:]])

    df = pd.concat([df[df.columns[:num_cols + 2]],binary_cats], axis = 1)

    churn_labels = df['label']

    features = df[df.columns[1:]]

    return np.array(features), np.array(churn_labels)


def split_train_val_data(features, labels, prop=0.5):
    """
    Split the features and labels sets deterministiclly into a training set
    and a validation set. The first <prop> of the dataset will be the
    training set, and the remaining data points will be the validation set.

    :param features:  a dataframe/numpy matrix-like feature set
    :param labels: a dataframe/numpy matrix-like label set
    :param prop: proportion of training set
    :return: train features, validation features, train labels, validation 
             labels
    """
    assert(features.shape[0] == labels.shape[0]), \
        "Dimension of features and label sets are not equal."

    num_train_data = int(features.shape[0] * prop)

    return features[0:num_train_data], \
           features[num_train_data:], \
           labels[0:num_train_data], \
           labels[num_train_data:]
