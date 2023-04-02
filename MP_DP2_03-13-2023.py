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
from skopt.space import Real, Integer
from functools import partial
import os
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier






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
    elif model_type == 'adaboost':
        base_classifier = DecisionTreeClassifier(max_depth = 1)
        clf = AdaBoostClassifier(estimator=base_classifier, n_estimators=int(C_values[0]), random_state = 42)
        
        
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
    elif model_type == 'adaboost':
        n_estimators_space = Integer(10, 200)
        learning_rate_space = Real(0.01, 2, prior = 'log-uniform')
        objective_fn = partial(train_and_evaluate_model, model_type, X_train= X_train, y_train = y_train, X_test = X_test, y_test = y_test )
        try:
            result = gp_minimize(objective_fn, [n_estimators_space, learning_rate_space], n_calls=20, random_state=42)
            best_n_estimators = result.x[0]
            best_learning_rate = result.x[1]
            best_score = -result.fun
        except:
            print(f'Error occurred during optimization for {model_type}')
            
        if best_n_estimators is not None and best_learning_rate is not None:
            test_score = train_and_evaluate_model(model_type, [best_n_estimators, best_learning_rate], X_train, y_train, X_test, y_test)

        return best_n_estimators, best_learning_rate, test_score
    
    
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
    print('SVM Score', svm_score * -1,'\n')
    
    lr_C = optimize_C_for_model('logistic regression', X_train, y_train, X_test, y_test)
    print('Best C for Logistic Regression:' , lr_C)
    # Logistic Regression   
    lg_score= train_and_evaluate_model('logistic regression', [lr_C], X_train, y_train, X_test, y_test)
    print('lg_score', lg_score * -1,'\n')
    
    ##Neural Networks   
    nn_relu_C = optimize_C_for_model('nn_relu', X_train, y_train, X_test, y_test)
    
    print('Best C for NN with ReLU:', nn_relu_C)
    nn_relu = run_nn_experiment(nn_relu_C, 'relu', X_train, y_train, X_test, y_test)
    print('nn relu score', nn_relu * -1,'\n')
    
    nn_tanh_C = optimize_C_for_model('nn_tanh', X_train, y_train, X_test, y_test)
    print('Best C for NN with tanh:', nn_tanh_C)
    nn_tanh = run_nn_experiment(nn_tanh_C, 'tanh', X_train, y_train, X_test, y_test)
    print('nn tanh score', nn_tanh * -1,'\n')
    
    nn_logistic_C = optimize_C_for_model('nn_logistic', X_train, y_train, X_test, y_test)
    print('Best C for NN with logistic:', nn_logistic_C)
    nn_log= run_nn_experiment(nn_logistic_C, 'logistic', X_train, y_train, X_test, y_test)
    print('nn log score', nn_log * -1,'\n')
    
    
    best_n_estimators, best_learning_rate, adaboost_score = optimize_C_for_model('adaboost', X_train, y_train, X_test, y_test)
    print(f'Best number of estimators for AdaBoost: {best_n_estimators}')
    print(f'Best learning rate for AdaBoost: {best_learning_rate}')
    print(f'AdaBoost Score: {adaboost_score * -1}\n')
    
    
    model_names = ['SVM', 'Log Regression', 'NN ReLU', 'NN Tanh', 'NN Logistic', 'AdaBoost']
    model_scores = [
    svm_score * -1,
    lg_score * -1,
    nn_relu * -1,
    nn_tanh * -1,
    nn_log * -1,
    adaboost_score * -1
    ]   

    plt.plot(model_names, model_scores, marker='o')
    plt.xlabel('Models')
    plt.ylabel('Test Accuracy')
    plt.title('Model Comparison')
    plt.grid()
    plt.xticks(rotation=45)
    plt.show()

    
if __name__ == '__main__':
    main()
