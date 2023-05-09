#Tensorflow/Keras
import tensorflow as tf
from sklearn import metrics
from sklearn import neural_network
from keras.models import Sequential
from keras import Input
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score

#Data Manipulation
import pandas as pd
import numpy as np

#Sklean
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, make_scorer
from sklearn.model_selection import GridSearchCV

#Models to be Compared
from sklearn.svm import SVC
from sklearn.naive_bayes import BernoulliNB
from sklearn import tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV

#Import Pre-Processed Data
from PreProcessFunctions import preProcess

#Visualization
import plotly
import plotly.express as px
import plotly.graph_objects as go 

# Ignore warnings
import warnings
warnings.filterwarnings('ignore')
tf.get_logger().setLevel('ERROR')

pd.options.display.max_columns=50

#Sets the random number to 0
np.random.seed(0)

pp = preProcess

#Assigning variables to the datasets
#X_train=pd.read_csv('Pre-Processed_X_Train_data.csv', encoding='utf-8')
#Y_train=pd.read_csv('Pre-Processed_Y_Train_data_copy.csv', encoding='utf-8')
#X_test=pd.read_csv('Pre-Processed_X_Test_data.csv', encoding='utf-8')
#y_test=pd.read_csv('Pre-Processed_Y_Test_data_copy.csv', encoding='utf-8')

#Select data for model
#X_train=dfX
#Y_train=dfY

class neuralNetwork(preProcess):
    #The creation of the FFNN model
    def create_network(self):
        model = Sequential(name="Model")
        model.add(Dense(units=14, input_dim=14, activation='softplus'))
        model.add(Dense(units=14, activation='relu'))
        model.add(Dense(units=1, activation='sigmoid'))

        model.compile(optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy'],
            )
        model.fit(preProcess.X_train,
              preProcess.Y_train,
              epochs=10,
              batch_size=10,
              verbose=2)
        return model
    
    def buildNetwork(self):
        #This Section is for the grid search 
        neural_network = KerasClassifier(build_fn=self.create_network,
                                         epochs=10,
                                         batch_size=10,
                                         #verbose=4
                                         )

        print(cross_val_score(neural_network, preProcess.X_train, preProcess.Y_train, cv =3))

    def gridSearch(self):
        batch_size = [10, 20, 40, 60, 80, 100]
        epochs = [10]
        #Define grid library
        param_grid = dict(batch_size=batch_size, epochs=epochs)
        scorers = {'precision_score': make_scorer(precision_score),
                   'recall_score': make_scorer(recall_score),
                   'accuracy_score': make_scorer(accuracy_score)}
        grid = GridSearchCV(estimator=neural_network, param_grid=param_grid, n_jobs=-1, cv=3, scoring=scorers,refit="precision_score")
        grid_result = grid.fit(preProcess.X_train, preProcess.Y_train)

        #Preparing for model reports
        y_pred = self.create_network.predict(preProcess.X_test)
        y_pred = (y_pred > 0.5)
        y_pred = y_pred.astype(int)
        cm = confusion_matrix(preProcess.y_test, y_pred)

        print()
        print('============================== Neural Network Confusion Matrix ==============================')
        print(cm)
        print()
    
        print()
        print('============================== Neural Network Accuracy Score ==============================')
        print('Accuracy:', accuracy_score(preProcess.y_test, y_pred))
        print()
    
        print()
        print('============================== Neural Network Classification Report ==============================')
        print(classification_report(preProcess.y_test, y_pred))
        print()
    
        #Remove "#" when grid search is needed
        print()
        print('============================== Neural Network GridSearch ==============================')
        print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
        print()

##Other models to compare
##Train KNeighborsClassifier Model
#KNN_Classifier = KNeighborsClassifier(n_jobs=-1)
#KNN_Classifier.fit(X_train, Y_train)
#
##Train LogisticRegression Model
#LGR_Classifier = LogisticRegression(n_jobs=-1, random_state=0)
#LGR_Classifier.fit(X_train, Y_train)
#
##Train Guassian Naive Baye Model
#BNB_Classifier = BernoulliNB()
#BNB_Classifier.fit(X_train, Y_train)
#
##Train Decision Tree Model
#DTC_Classifier = tree.DecisionTreeClassifier(criterion='entropy', random_state=0)
#DTC_Classifier.fit(X_train, Y_train)
#
#models = []
#models.append(('Naive Baye Classifier', BNB_Classifier))
#for i, v in models:
#    scores = cross_val_score(v, X_train, Y_train, cv = 10)
#    accuracy = metrics.accuracy_score(Y_train, v.predict(X_train))
#    confusion_matrix = metrics.confusion_matrix(Y_train, v.predict(X_train))
#    classification = metrics.classification_report(Y_train, v.predict(X_train))
#
#print()
#print('============================== {} Model Evaluation =============================='.format(i))
#print()
#print ("Cross Validation Mean Score:" "\n", scores.mean())
#print()
#print ("Model Accuracy:" "\n", accuracy)
#print()
#print("Confusion matrix:" "\n", confusion_matrix)
#print()
#print("Classification report:" "\n", classification) 
#print()
#
#models.append(('Decision Tree Classifier', DTC_Classifier))
#for i, v in models:
#    scores = cross_val_score(v, X_train, Y_train, cv = 10)
#    accuracy = metrics.accuracy_score(Y_train, v.predict(X_train))
#    confusion_matrix = metrics.confusion_matrix(Y_train, v.predict(X_train))
#    classification = metrics.classification_report(Y_train, v.predict(X_train))
#
#print()
#print('============================== {} Model Evaluation =============================='.format(i))
#print()
#print ("Cross Validation Mean Score:" "\n", scores.mean())
#print()
#print ("Model Accuracy:" "\n", accuracy)
#print()
#print("Confusion matrix:" "\n", confusion_matrix)
#print()
#print("Classification report:" "\n", classification) 
#print()
#
#models.append(('KNeighborsClassifier', KNN_Classifier))
#
#for i, v in models:
#    scores = cross_val_score(v, X_train, Y_train, cv = 10)
#    accuracy = metrics.accuracy_score(Y_train, v.predict(X_train))
#    confusion_matrix = metrics.confusion_matrix(Y_train, v.predict(X_train))
#    classification = metrics.classification_report(Y_train, v.predict(X_train))
#
#print()
#print('============================== {} Model Evaluation =============================='.format(i))
#print()
#print ("Cross Validation Mean Score:" "\n", scores.mean())
#print()
#print ("Model Accuracy:" "\n", accuracy)
#print()
#print("Confusion matrix:" "\n", confusion_matrix)
#print()
#print("Classification report:" "\n", classification) 
#print()
#models.append(('LogisticRegression', LGR_Classifier))
#
#for i, v in models:
#    scores = cross_val_score(v, X_train, Y_train, cv = 10)
#    accuracy = metrics.accuracy_score(Y_train, v.predict(X_train))
#    confusion_matrix = metrics.confusion_matrix(Y_train, v.predict(X_train))
#    classification = metrics.classification_report(Y_train, v.predict(X_train))
#
#print()
#print('============================== {} Model Evaluation =============================='.format(i))
#print()
#print ("Cross Validation Mean Score:" "\n", scores.mean())
#print()
#print ("Model Accuracy:" "\n", accuracy)
#print()
#print("Confusion matrix:" "\n", confusion_matrix)
#print()
#print("Classification report:" "\n", classification) 
#print()



#means = grid_result.cv_results_['mean_test_score']
#stds = grid_result.cv_results_['std_test_score']
#params = grid_result.cv_results_['params']
#for mean, stdev, param in zip(means, stds, params):
#    print("%f (%f) with: %r" % (mean, stdev, param))

