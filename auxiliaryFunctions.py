import numpy as np
import pandas as pd
from sklearn import metrics
from joblib import dump, load

import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.compat.v1.disable_v2_behavior()
tf.compat.v1.disable_eager_execution()


#region Auxiliary functions

def getMetrics(config_str, y, yhat):
    model_metrics = dict()
    model_metrics['model'] = config_str
    model_metrics['accuracy'] = np.round(metrics.accuracy_score(y, yhat), 4)
    model_metrics['error'] = 1 - model_metrics['accuracy']
    model_metrics['precision'] = np.round(metrics.precision_score(y, yhat), 4)
    model_metrics['recall'] = np.round(metrics.recall_score(y, yhat), 4)
    model_metrics['f1-score'] = np.round(metrics.f1_score(y, yhat), 4)

    return pd.DataFrame(data = model_metrics, index = [0])


def writeModelResults(results, model, model_type):
    
    results.to_csv('./results/model_results.csv', mode = 'a', header = False, index = False)
    model_path = './models/' + results['model'].values[0]
    if model_type == 'decisionTree' or model_type == 'SVM':
        model_path += '.joblib'
        dump(model, model_path) 
    elif model_type == 'NaiveBayes':
        model_path += '.csv'
        model.to_csv(model_path)
    elif model_type == 'regLog':
        model_path += '.npy'
        model.dump(model_path)

    print('Model results saved succesfully')
    return 


def readModels(model_name, model_type):

    model_path = './models/' + model_name
    if model_type == 'decisionTree' or model_type == 'SVM':
        model_path += '.joblib'
        model = load(model_path)
    elif model_type == 'NaiveBayes':
        model_path += '.csv'
        model = pd.read_csv(model_path, index_col = [0],  header = [0,1])
    elif model_type == 'regLog':
        model_path += '.npy'
        model = np.load(model_path, allow_pickle = True)

    return model


#endregion


#region Predictive functions

def get_logRegPreds(x_predict, weights, return_probs = False):

    #Initialize the graph
    tf.reset_default_graph()

    #Defining tensors and variables 
    X = tf.placeholder(dtype = tf.float32, shape = [None, x_predict.shape[1] + 1], name = 'features') 
    W = tf.placeholder(dtype = tf.float32, shape = [x_predict.shape[1] + 1, 1], name = 'weights')

    #Estimating values
    Z = tf.matmul(X, W, name = 'logit')
    probs = tf.nn.sigmoid(Z, name = 'probs')
    preds = tf.cast(tf.round(probs), tf.int32, name = 'predictions')

    with tf.Session() as session:

        #Initialize global vars
        session.run(tf.global_variables_initializer())

        #Tensorboard writer
        writer = tf.summary.FileWriter('./graphs/regLog_predictions', session.graph)

        #Reshaping data
        ones = np.expand_dims(np.ones_like(x_predict.values[:,0]), axis = 1)
        x = np.hstack((ones, x_predict.values))

        #Whole batch dictionary
        feed_dict = {X:x, W:weights}
        
        #Making the predictions
        predictions, probabilities = session.run([preds, probs], feed_dict = feed_dict)

        #Closing the writer
        writer.close()

        if return_probs:
            return np.squeeze(probabilities)
        else:
            return np.squeeze(predictions, axis = 1)


def getBayesProb(row, model, return_probs = False):
    #Retrive metrics
    class_probs = model.iloc[:,-1]
    summary = model.iloc[:,:-1]

    #Calculate probs
    row_prob = dict() 
    for cat, classProb in class_probs.items():
        prob = classProb   
        for index, value in row.items():
            xstd = summary.loc[cat, (index,'std')]
            xmean = summary.loc[cat, (index,'mean')]
            prob *= (1/(np.sqrt(2 * np.pi) * xstd)) * np.exp(-np.power(value - xmean, 2)/(2 * np.power(xstd,2)))
        row_prob[cat] = prob

    max_class = max(row_prob, key = row_prob.get)

    if return_probs:
        return np.array(list(row_prob.values()))
    else:
        return max_class


def getModelPredictions(model, x_predict, model_type, return_probs = False):
    if model_type == 'decisionTree' or model_type == 'SVM':
        y_pred = model.predict(x_predict)
    elif model_type == 'NaiveBayes':
        y_pred = x_predict.apply(lambda row: getBayesProb(row, model, return_probs), axis = 1)
        if return_probs:
            y_pred = np.reshape(np.concatenate(y_pred.values), (-1, 2))
        else:
            y_pred = y_pred.values
    elif model_type == 'regLog':
        y_pred = get_logRegPreds(x_predict, model, return_probs)

    return y_pred

#endregion


#region Ensemble functions

def get_topModels(df):
    top_models = df.loc[df.groupby(['model_type'])['accuracy'].idxmax()]
    top_models.set_index('model_type', inplace = True)
    return top_models


def ensemblePredictions(top_models, X_test):
    #Read and generate all top model predictions
    predictions = dict()
    for index, row in top_models.iterrows():
        model = readModels(row['model'], index)
        predictions[index] = getModelPredictions(model, X_test, model_type = index)

    #Generate mode model
    preds = np.array(list(predictions.values()))
    means = np.mean(preds, axis = 0)
    #mode = np.rint(np.nextafter(means, means+1))
    mode = np.round(means)
    mode = mode.astype(int)
    predictions['mode'] = mode
    df = pd.DataFrame(data = predictions)

    return  df


def ensembleMetrics(predictions_df, y):
    model_metrics = pd.DataFrame(columns=['model', 'accuracy', 'error', 'precision', 'recall', 'f1-score'])
    for model_name in predictions_df:
        df = getMetrics(model_name, y, predictions_df[model_name].values)
        model_metrics = model_metrics.append(df, ignore_index = True)

    return model_metrics

#endregion