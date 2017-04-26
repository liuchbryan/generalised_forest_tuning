import numpy as np
from datetime import datetime
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier


def get_RF_performance(RF_model,
                       X_train, X_val, y_train, y_val,
                       runs=3, verbose=False):
    # Provision to store the AUC, the root mean squared difference
    # in successive predictions (RMS-predicition delta), and runtime
    # for model training
    auc = np.zeros(runs)
    pred_proba = np.zeros(shape=(runs, X_val.shape[0]))
    rms_pred_delta = np.zeros(runs)
    runtime = np.zeros(runs)
    
    # Run experiment 'runs' times
    for i in range(0, runs):
        # Measure the time taken for model to train
        train_start_time = datetime.now()
        RF_model.fit(X_train, y_train)
        train_end_time = datetime.now()
        
        pos_class_proba = RF_model.predict_proba(X_val)[:, 1]
        
        # Get the AUC, and runtime for this run,
        # and store the prediction probabilities for calculating MSPD
        auc[i] = roc_auc_score(y_val, pos_class_proba)
        runtime[i] = (train_end_time - train_start_time).total_seconds()
        pred_proba[i] = pos_class_proba
    
    # Calculate the MSPD
    mspd_acc = 0
    for i in range(0, runs):
        for j in range(0, i):
            mspd_acc += 2.0 / (runs * (runs - 1)) * \
                np.mean(
                        np.power(pred_proba[i] - pred_proba[j],
                                 2)
                        )

    # Collate result
    result = {}
    result['AUC'] = np.mean(auc)
    result['AUC stdev'] = np.std(auc, ddof=1)
    result['RMSPD'] = np.sqrt(mspd_acc)
    result['MSPD'] = mspd_acc
    result['Pred Var'] = np.mean(np.var(pred_proba, axis = 1))
    result['Pred Stdev'] = np.sqrt(np.mean(np.var(pred_proba, axis = 1)))
    result['Runtime'] = np.mean(runtime)
    result['Runtime stdev'] = np.std(runtime, ddof=1)
    
    return result


def train_and_get_RF_performance(params,
                                 features_train_complete,
                                 features_val,
                                 labels_train_complete,
                                 labels_val,
                                 nrun=3):
    
    # Randomly select n available training samples for actual training
    # n = total available training sample * proportion selected (=param[2])
    train_sample_index = \
        np.random.randint(0, features_train_complete.shape[0],
                          np.int64(features_train_complete.shape[0]*params[2]))
    
    features_train = features_train_complete[train_sample_index]
    labels_train = labels_train_complete[train_sample_index]
    
    RF_model = RandomForestClassifier(n_estimators=np.int64(params[0]),
                                      max_depth=np.int64(params[1]),
                                      n_jobs=-1)

    return get_RF_performance(RF_model, 
                              features_train, features_val, 
                              labels_train, labels_val, 
                              nrun)


def get_RF_generalised_performance_score(params,
                                         features_train_complete,
                                         features_val,
                                         labels_train_complete,
                                         labels_val,
                                         weight_alpha=1,
                                         weight_beta=1,
                                         weight_gamma=1,
                                         nrun=3,
                                         verbose=False):
    
    """
    Runs and calculate the generalised performance score for a random
    forest under the given parameters.
        
    The generalised performance score is a weighted sum between:
    - AUC, weighted by 'weight_alpha':
    - Negative RMSPD (root mean squared prediction delta for 
      successive pred.), weighted by 'weight_beta':
    - Negative Runtime (training time in seconds), weighted by 
      'weight_gamma':
    """
    
    this_performance = train_and_get_RF_performance(
        params, features_train_complete, features_val, 
        labels_train_complete, labels_val, nrun)
                                                    
    if verbose:
        print(this_performance)
    
    # Return the negative loss function, as pybo *maximises* the
    # posterior mean
    return weight_alpha * this_performance['AUC'] - \
           weight_beta * this_performance['RMSPD'] - \
           weight_gamma * this_performance['Runtime']
