import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, precision_recall_fscore_support, accuracy_score
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.neural_network import MLPClassifier
import numpy as np
from skopt import gp_minimize
from skopt.space import Real
from functools import partial
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import warnings
warnings.filterwarnings("ignore", category=UserWarning)




    
def train_and_evaluate_model(model_type, C_values, X_train, y_train, X_test, y_test):
    if model_type == 'svm':
        clf = svm.SVC(C=C_values[0], kernel='linear')
    elif model_type == 'logistic regression':
        clf = LogisticRegression(C=C_values[0], solver='lbfgs')
    elif model_type == 'nn_relu':
        nn = MLPClassifier(hidden_layer_sizes=(10, 10), activation='relu', alpha=C_values[0], max_iter=1000)
        nn.fit(X_train, y_train.ravel())
        clf = nn
    elif model_type == 'nn_tanh':
        nn = MLPClassifier(hidden_layer_sizes=(10, 10), activation='tanh', alpha=C_values[0], max_iter=1000)
        nn.fit(X_train, y_train.ravel())
        clf = nn
    elif model_type == 'nn_logistic':
        nn = MLPClassifier(hidden_layer_sizes=(10, 10), activation='logistic', alpha=C_values[0], max_iter=1000)
        nn.fit(X_train, y_train.ravel())
        clf = nn
    else:
        raise ValueError(f'Unknown model type: {model_type}')

    clf.fit(X_train, y_train.ravel())
    train_acc = clf.score(X_train, y_train)
    test_acc = clf.score(X_test, y_test)
    return -test_acc 

    
    

def run_nn_experiment(C_values, activation, X_train, y_train, X_test, y_test):
    # convert C_values to a list if it's not already
    if not isinstance(C_values, list):
        C_values = [C_values]
    
    # perform grid search over the specified values of C
    best_score = -1
    for C in C_values:
        nn = MLPClassifier(hidden_layer_sizes=(10,10), alpha=C, activation=activation, solver='adam', max_iter=1000, random_state=42)
        nn.fit(X_train, y_train.ravel())
        score = nn.score(X_test, y_test)
        if score > best_score:
            best_score = score
            best_nn = nn
    
    #print(f'Best C for neural network with {activation} activation: {best_nn.alpha:.3g}, best test accuracy: {best_score:.3f}\n')
    return -best_score 

    
def run_cnn(activation, X_train, y_train, X_test, y_test):
    cnn = models.Sequential()
    cnn.add(layers.Conv1D(32, 3, activation=activation, input_shape=(19, 1)))
    cnn.add(layers.MaxPooling1D(2))
    cnn.add(layers.Conv1D(64, 3, activation=activation))
    cnn.add(layers.MaxPooling1D(2))
    cnn.add(layers.Conv1D(64, 3, activation=activation))
    cnn.add(layers.Flatten())
    cnn.add(layers.Dense(64, activation=activation))
    cnn.add(layers.Dense(10))
    cnn.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
    cnn.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))
    test_loss, test_acc = cnn.evaluate(X_test,  y_test, verbose=2)
    
    return test_acc



def optimize_C_for_model(model_type, X_train, y_train, X_test, y_test):
    C_values = [1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100]
    C_values_with_extra = C_values + [1000]
    best_C = None

    if model_type == 'svm':
        C_space = Real(1e-7, 100, prior='log-uniform')
        objective_fn = partial(train_and_evaluate_model, 'svm', X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test)
    elif model_type == 'logistic regression':
        C_space = Real(1e-7, 100, prior='log-uniform')
        objective_fn = partial(train_and_evaluate_model, 'logistic regression', X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test)
    elif model_type == 'nn_relu':
        C_space = Real(1e-7, 1000, prior='log-uniform')
        objective_fn = partial(run_nn_experiment, activation='relu', X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test)
    elif model_type == 'nn_tanh':
        C_space = Real(1e-7, 1000, prior='log-uniform')
        objective_fn = partial(run_nn_experiment, activation='tanh', X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test)
    elif model_type == 'nn_logistic':
        C_space = Real(1e-7, 1000, prior='log-uniform')
        objective_fn = partial(run_nn_experiment, activation='logistic', X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test)
    else:
        raise ValueError(f'Unknown model type: {model_type}')

    try:
        result = gp_minimize(objective_fn, [C_space], n_calls=20, random_state=42)
        best_C = result.x[0]
        best_score = -result.fun
        #print(f'Best C for {model_type}: {best_C:.3g}, best test accuracy: {best_score:.3f}\n')
    except:
        print(f'Error occurred during optimization for {model_type}')
    
    if best_C is not None:
        test_score = train_and_evaluate_model(model_type, [best_C], X_train, y_train, X_test, y_test)
    
    return best_C




def main():
    df = pd.read_csv('clean_dataset.csv')


    one_hot1 = pd.get_dummies(df['Ethnicity'])
    one_hot2 = pd.get_dummies(df['Citizen'])
    
    df.drop(['Ethnicity', 'Industry', 'ZipCode', 'Citizen'], axis=1, inplace=True)
    df = df.join(one_hot1)
    df = df.join(one_hot2)
    
    y = df['Approved'].to_numpy()
    df.drop(['Approved'], axis=1, inplace=True)
    X = df.to_numpy()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
    
    X_train = preprocessing.scale(X_train)
    X_test = preprocessing.scale(X_test)

 
    
    svm_C = optimize_C_for_model('svm', X_train, y_train, X_test, y_test)
    print('Best C for SVM:', svm_C)
    
    
    # Support Vector Machines
    svm_score = train_and_evaluate_model('svm', [svm_C], X_train, y_train, X_test, y_test)
    print('SVM Score', svm_score * -1)
    
    lr_C = optimize_C_for_model('logistic regression', X_train, y_train, X_test, y_test)
    print('Best C for Logistic Regression:' , lr_C)
    # Logistic Regression   
    lg_score= train_and_evaluate_model('logistic regression', [lr_C], X_train, y_train, X_test, y_test)
    print('lg_score', lg_score * -1)
    
    ##Neural Networks   
    nn_relu_C = optimize_C_for_model('nn_relu', X_train, y_train, X_test, y_test)
    
    print('Best C for NN with ReLU:', nn_relu_C)
    nn_relu = run_nn_experiment(nn_relu_C, 'relu', X_train, y_train, X_test, y_test)
    print('nn relu score', nn_relu * -1)
    
    nn_tanh_C = optimize_C_for_model('nn_tanh', X_train, y_train, X_test, y_test)
    print('Best C for NN with tanh:', nn_tanh_C)
    nn_tanh = run_nn_experiment(nn_tanh_C, 'tanh', X_train, y_train, X_test, y_test)
    print('nn tanh score', nn_tanh * -1)
    
    nn_logistic_C = optimize_C_for_model('nn_logistic', X_train, y_train, X_test, y_test)
    print('Best C for NN with logistic:', nn_logistic_C)
    nn_log= run_nn_experiment(nn_logistic_C, 'logistic', X_train, y_train, X_test, y_test)
    print('nn log score', nn_log * -1)
    
    cnn = run_cnn('relu', X_train, y_train, X_test, y_test)
    print('cnn relu score', cnn)
    

main()   
