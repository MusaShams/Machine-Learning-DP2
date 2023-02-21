import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, precision_recall_fscore_support, accuracy_score
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve
from sklearn import svm
from sklearn import preprocessing
from sklearn.neural_network import MLPClassifier
from sklearn.neural_network import MLPRegressor






def train_and_evaluate_svms(C, X_train, y_train, X_test, y_test):
    score_list_svm = []
    score_list_svm_rbf = []
    score_list_svm_poly = []


    for i in C:
        model_svm= svm.SVC(kernel='linear', C=i)
        model_svm_rbf= svm.SVC(kernel='rbf', C=i)
        model_svm_poly= svm.SVC(kernel='poly',degree=2, C=i)
       
        model_svm.fit(X_train,y_train.ravel())
        model_svm_rbf.fit(X_train,y_train.ravel())
        model_svm_poly.fit(X_train,y_train.ravel())
        
        score_svm = model_svm.score(X_train, y_train)
        score_svm_rbf = model_svm_rbf.score(X_train, y_train)
        score_svm_poly = model_svm_poly.score(X_train, y_train)

        score_list_svm.append(score_svm)
        score_list_svm_rbf.append(score_svm_rbf)
        score_list_svm_poly.append(score_svm_poly)
        
        
    print("SVM Training accuracy scores:")
    for i in range(0, len(C)):
        print(f'C Value {C[i]:.3g}: {score_list_svm[i]:.3g}')  
        print(f'C Value {C[i]:.3g}: {score_list_svm_rbf[i]:.3g}')   
        print(f'C Value {C[i]:.3g}: {score_list_svm_poly[i]:.3g}')
    print('\n\n')
        
        

  
    #print("(Train) Linear Scores:" , score_list_svm, '\n')
    #print("(Train) RBF Scores:" , score_list_svm, '\n')
    #print("(Train) Poly Scores:" , score_list_svm, '\n')
    
    
    score_list_svm_test = []
    score_list_svm_rbf_test = []
    score_list_svm_poly_test = []
    
    for i in C:
        model_svm= svm.SVC(kernel='linear', C=i)
        model_svm_rbf= svm.SVC(kernel='rbf', C=i)
        model_svm_poly= svm.SVC(kernel='poly',degree=2, C=i)
       
        model_svm.fit(X_test,y_test.ravel())
        model_svm_rbf.fit(X_test,y_test.ravel())
        model_svm_poly.fit(X_test,y_test.ravel())
        
        score_svm = model_svm.score(X_test, y_test)
        score_svm_rbf = model_svm_rbf.score(X_test, y_test)
        score_svm_poly = model_svm_poly.score(X_test, y_test)

        score_list_svm_test.append(score_svm)
        score_list_svm_rbf_test.append(score_svm_rbf)
        score_list_svm_poly_test.append(score_svm_poly)
        
    #print("(Test) Linear Scores:" , score_list_svm_test, '\n')
    #print("(Test) RBF Scores:" , score_list_svm_test, '\n')
    #print("(Test) Poly Scores:" , score_list_svm_test, '\n')
    
    print("SVM Test accuracy scores:")
    for i in range(0, len(C)):
        print(f'C Value {C[i]:.3g}: {score_list_svm_test[i]:.3g}')  
        print(f'C Value {C[i]:.3g}: {score_list_svm_rbf_test[i]:.3g}')   
        print(f'C Value {C[i]:.3g}: {score_list_svm_poly_test[i]:.3g}')
    print('\n\n')
        


    
def logistic_regression(X_train, y_train, X_test, y_test, C):
    score_list_l2 = []
    score_list_l1 = []
    score_list_elasticnet = []

    for i in C:
        model_l2= LogisticRegression(penalty = 'l2' , max_iter=3000, C=i)
        model_l1= LogisticRegression(penalty = 'l1' , max_iter=3000, solver='liblinear', C= i )
        model_elasticnet= LogisticRegression(penalty = 'elasticnet' , max_iter=30000, solver= 'saga', l1_ratio=1, C=i)

        model_l2.fit(X_train,y_train)
        model_l1.fit(X_train,y_train)
        model_elasticnet.fit(X_train,y_train)

        score_l2 = model_l2.score(X_train, y_train)
        score_l1 = model_l1.score(X_train, y_train)
        score_elasticnet = model_elasticnet.score(X_train, y_train)

        score_list_l2.append(score_l2)
        score_list_l1.append(score_l1)
        score_list_elasticnet.append(score_elasticnet)
        
    #print("(Train) L2 Scores:" , score_list_l2, '\n')
    #print("(Train) L1 Scores:" , score_list_l1, '\n')
    #print("(Train) Elastic Net Scores:" , score_list_elasticnet, '\n')
    
    
    print("Logistic Regression Training accuracy scores:")
    for i in range(0, len(C)):
        print(f'C Value {C[i]:.3g}: {score_list_l2[i]:.3g}')  
        print(f'C Value {C[i]:.3g}: {score_list_l1[i]:.3g}')   
        print(f'C Value {C[i]:.3g}: {score_list_elasticnet[i]:.3g}')
    print('\n\n')

    score_list_l2_test = []
    score_list_l1_test = []
    score_list_elasticnet_test = []

    for i in C:
        model_l2= LogisticRegression(penalty = 'l2' , max_iter=3000, C=i)
        model_l1= LogisticRegression(penalty = 'l1' , max_iter=3000, solver='liblinear', C= i )
        model_elasticnet= LogisticRegression(penalty = 'elasticnet' , max_iter=30000, solver= 'saga', l1_ratio=1, C=i)

        model_l2.fit(X_test,y_test)
        model_l1.fit(X_test,y_test)
        model_elasticnet.fit(X_test,y_test)

        score_l2 = model_l2.score(X_test,y_test)
        score_l1 = model_l1.score(X_test,y_test)
        score_elasticnet = model_elasticnet.score(X_test,y_test)

        score_list_l2_test.append(score_l2)
        score_list_l1_test.append(score_l1)
        score_list_elasticnet_test.append(score_elasticnet)
        
    #print("(Test) L2 Scores:" , score_list_l2_test, '\n')
    #print("(Test)) L1 Scores:" , score_list_l1_test, '\n')
    #print("(Test)) Elastic Net Scores:" , score_list_elasticnet_test, '\n')
    
    
    print("Logistic Regression Test accuracy scores:")
    for i in range(0, len(C)):
        print(f'C Value {C[i]:.3g}: {score_list_l2_test[i]:.3g}')   
        print(f'C Value {C[i]:.3g}: {score_list_l1_test[i]:.3g}')       
        print(f'C Value {C[i]:.3g}: {score_list_elasticnet_test[i]:.3g}')
    print('\n\n')
    
        
        





def run_nn_experiment(C_vals, activation, X_train, y_train, X_test, y_test):
    acc_train = []
    acc_test = []
    for C in C_vals:
        nn = MLPClassifier(hidden_layer_sizes=(50), activation=activation, solver='adam', max_iter=10000, alpha=C)
        nn.fit(X_train, y_train.ravel())
        y_hat_train_nn = nn.predict(X_train)
        acc_train.append(accuracy_score(y_train, y_hat_train_nn))
        y_hat_test_nn = nn.predict(X_test)
        acc_test.append(accuracy_score(y_test, y_hat_test_nn))
        cm = confusion_matrix(y_test, y_hat_test_nn)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=nn.classes_)
        title = "Accuracy Score" 
        disp.plot()
        disp.ax_.set_title(title)
        plt.xlabel(f"Predicted label\nHidden Layer: 50, Activation: {activation}, Iters: 1000, C: {C}\n")
        plt.show()
        
    print(""f"Neural Networks {activation} Training accuracy scores:")
    for i in range(0, len(C_vals)):
        print(f'C Value {C_vals[i]:.3g}: {acc_train[i]:.3g}')
    print('\n\n')
    print(""f"Neural Networks {activation} Test accuracy scores:")
    for i in range(0, len(C_vals)):
        print(f'C Value {C_vals[i]:.3g}: {acc_test[i]:.3g}')
    print('\n\n')



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
    

    C_NN = [0.0000001, 0.00001, 0.001, 0.01, 0.1, 1, 10, 100, 10000, 1000000]   
    C = [0.0000001, 0.00001, 0.001, 0.01, 0.1, 1, 10, 100, 10000]   
    
    
    ##SVM
    train_and_evaluate_svms(C, X_train, y_train, X_test, y_test)
    
    ##Logistic Regression   
    logistic_regression(X_train, y_train, X_test, y_test, C)
    
    
    ##Neural Networks   
    run_nn_experiment(C_NN, 'relu', X_train, y_train, X_test, y_test)
    run_nn_experiment(C_NN, 'tanh', X_train, y_train, X_test, y_test)
    run_nn_experiment(C_NN, 'logistic', X_train, y_train, X_test, y_test)
    
    
    



main()   
