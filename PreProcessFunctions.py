# import relevant modules
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
#import seaborn as sns
import sklearn as sns
#import imblearn

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
#from PreProcessFunctions import preProcess

#Visualization
import plotly
import plotly.express as px
import plotly.graph_objects as go 
from tkinter import messagebox, filedialog, Toplevel, Entry, Button
import queue
from threading import *

# Ignore warnings
import warnings
warnings.filterwarnings('ignore')

##Data assigned variables and attributes removed
#def readTrainCSV():
#    filename = filedialog.askopenfilename(initialdir= "F:\Downloads\Knowledge\Programming\Self-Projects\Python Projects\Windowed Neural Network\Data", title="Select Training Data", filetypes=(("Comma-Seperated Value (*.csv*)", ("*.csv*")),("Text Files (*.txt*)", "*.txt*"), ("All Files (*.*)", "*.*")))
#    train = pd.read_csv(filename)
#    train.drop(['num_outbound_cmds','dst_host_srv_rerror_rate','dst_host_rerror_rate','dst_host_srv_serror_rate','dst_host_serror_rate','dst_host_srv_diff_host_rate','dst_host_same_src_port_rate','dst_host_diff_srv_rate','dst_host_same_srv_rate','dst_host_srv_count','dst_host_count','srv_diff_host_rate','diff_srv_rate','same_srv_rate','srv_rerror_rate','rerror_rate','srv_serror_rate','num_shells','num_root','root_shell','num_compromised','hot','urgent','land','wrong_fragment','count','srv_count'], axis=1, inplace=True)
#    return train
#
#def readTestCSV():
#    filename = filedialog.askopenfilename(initialdir= "F:\Downloads\Knowledge\Programming\Self-Projects\Python Projects\Windowed Neural Network\Data", title="Select Testing Data", filetypes=(("Comma-Seperated Value (*.csv*)", ("*.csv*")),("Text Files (*.txt*)", "*.txt*"), ("All Files (*.*)", "*.*")))
#    test = pd.read_csv(filename)
#    test.drop(['num_outbound_cmds','dst_host_srv_rerror_rate','dst_host_rerror_rate','dst_host_srv_serror_rate','dst_host_serror_rate','dst_host_srv_diff_host_rate','dst_host_same_src_port_rate','dst_host_diff_srv_rate','dst_host_same_srv_rate','dst_host_srv_count','dst_host_count','srv_diff_host_rate','diff_srv_rate','same_srv_rate','srv_rerror_rate','rerror_rate','srv_serror_rate','num_shells','num_root','root_shell','num_compromised','hot','urgent','land','wrong_fragment','count','srv_count'], axis=1, inplace=True)
#    return test
#
#
#train = readTrainCSV()
#test = readTestCSV()
#from sklearn.preprocessing import StandardScaler
#scaler = StandardScaler()
#
## extract numerical attributes and scale it to have zero mean and unit variance  
#cols = train.select_dtypes(include=['float64','int64']).columns
#sc_train = scaler.fit_transform(train.select_dtypes(include=['float64','int64']))
#sc_test = scaler.fit_transform(test.select_dtypes(include=['float64','int64']))
#
## turn the result back to a dataframe
#sc_traindf = pd.DataFrame(sc_train, columns = cols)
#sc_testdf = pd.DataFrame(sc_test, columns = cols)
#
#from sklearn.preprocessing import LabelEncoder
#encoder = LabelEncoder()
#
#
## extract categorical attributes from both training and test sets 
#cattrain = train.select_dtypes(include=['object']).copy()
#cattest = test.select_dtypes(include=['object']).copy()
#
## encode the categorical attributes
#traincat = cattrain.apply(encoder.fit_transform)
#testcat = cattest.apply(encoder.fit_transform)
#
## separate target column from encoded data 
#enctrain = traincat.drop(['class'], axis=1)
#cat_Ytrain = traincat[['class']].copy()
#
#train_x = pd.concat([sc_traindf,enctrain],axis=1)
#train_y = train['class']
#print(train_x.shape)
#
#test_df = pd.concat([sc_testdf,testcat],axis=1)
#print(test_df.shape)
#
#from sklearn.preprocessing import LabelEncoder
#encoder = LabelEncoder()
#
## extract categorical attributes from both training and test sets 
#cattrain = train.select_dtypes(include=['object']).copy()
#cattest = test.select_dtypes(include=['object']).copy()
#
## encode the categorical attributes
#traincat = cattrain.apply(encoder.fit_transform)
#testcat = cattest.apply(encoder.fit_transform)
#
## separate target column from encoded data 
#enctrain = traincat.drop(['class'], axis=1)
#cat_Ytrain = traincat[['class']].copy()
#
#train_x = pd.concat([sc_traindf,enctrain],axis=1)
#train_y = train['class']
#print(train_x.shape)
#
#test_df = pd.concat([sc_testdf,testcat],axis=1)
#print(test_df.shape)
#
#from sklearn.ensemble import RandomForestClassifier
#rfc = RandomForestClassifier();
#
## fit random forest classifier on the training set
#rfc.fit(train_x, train_y);
## extract important features
#score = np.round(rfc.feature_importances_,3)
#importances = pd.DataFrame({'feature':train_x.columns,'importance':score})
#importances = importances.sort_values('importance',ascending=False).set_index('feature')
## plot importances
#plt.rcParams['figure.figsize'] = (11, 4)
#importances.plot.bar();
#
#from sklearn.feature_selection import RFE
#import itertools
#rfc = RandomForestClassifier()
#
## create the RFE model and select 10 attributes
#rfe = RFE(rfc, n_features_to_select=15)
#rfe = rfe.fit(train_x, train_y)
#
## summarize the selection of the attributes
#feature_map = [(i, v) for i, v in itertools.zip_longest(rfe.get_support(), train_x.columns)]
#selected_features = [v for i, v in feature_map if i==True]
#
##print(selected_features)
#
#from sklearn.model_selection import train_test_split
#X_train,X_test,Y_train,Y_test = train_test_split(train_x,train_y,train_size=0.70, random_state=2)
#
##Export pre-processed data into csv file
#X_train.to_csv(r"F:\Downloads\Knowledge\Programming\Self-Projects\Python Projects\Windowed Neural Network\Data\Pre-Processed_X_Train_data.csv", index = False)
#Y_train.to_csv(r"F:\Downloads\Knowledge\Programming\Self-Projects\Python Projects\Windowed Neural Network\Data\Pre-Processed_Y_Train_data.csv", index = False)
#X_test.to_csv(r"F:\Downloads\Knowledge\Programming\Self-Projects\Python Projects\Windowed Neural Network\Data\Pre-Processed_X_Test_data.csv", index = False)
#Y_test.to_csv(r"F:\Downloads\Knowledge\Programming\Self-Projects\Python Projects\Windowed Neural Network\Data\Pre-Processed_Y_Test_data.csv", index = False)

#PreProcessing    
class preProcess(object):
    

    #Data Selection Functions
    def readTrainCSV(self):
        filename = filedialog.askopenfilename(initialdir= "F:\Downloads\Knowledge\Programming\Self-Projects\Python Projects\Windowed Neural Network\Data", title="Select Training Data", filetypes=(("Comma-Seperated Value (*.csv*)", ("*.csv*")),("Text Files (*.txt*)", "*.txt*"), ("All Files (*.*)", "*.*")))
        self.train = pd.read_csv(filename)
        self.train.drop(['num_outbound_cmds','dst_host_srv_rerror_rate','dst_host_rerror_rate','dst_host_srv_serror_rate','dst_host_serror_rate','dst_host_srv_diff_host_rate','dst_host_same_src_port_rate','dst_host_diff_srv_rate','dst_host_same_srv_rate','dst_host_srv_count','dst_host_count','srv_diff_host_rate','diff_srv_rate','same_srv_rate','srv_rerror_rate','rerror_rate','srv_serror_rate','num_shells','num_root','root_shell','num_compromised','hot','urgent','land','wrong_fragment','count','srv_count'], axis=1, inplace=True)
        #Change Label Contents
        print("File Opened: " +filename)
        #return train

    def thread1(self):
        t2 = Thread(target=self.readTrainCSV)
        t2.start()
        #t2.join(30)
        if t2.is_alive():
            print("Test Thread is still active")
        else:
            print("Test Thread has stopped")
        t2.join    

    def readTestCSV(self):
        filename1 = filedialog.askopenfilename(initialdir= "F:\Downloads\Knowledge\Programming\Self-Projects\Python Projects\Windowed Neural Network\Data", title="Select Testing Data", filetypes=(("Comma-Seperated Value (*.csv*)", ("*.csv*")),("Text Files (*.txt*)", "*.txt*"), ("All Files (*.*)", "*.*")))
        self.test = pd.read_csv(filename1)
        self.test.drop(['num_outbound_cmds','dst_host_srv_rerror_rate','dst_host_rerror_rate','dst_host_srv_serror_rate','dst_host_serror_rate','dst_host_srv_diff_host_rate','dst_host_same_src_port_rate','dst_host_diff_srv_rate','dst_host_same_srv_rate','dst_host_srv_count','dst_host_count','srv_diff_host_rate','diff_srv_rate','same_srv_rate','srv_rerror_rate','rerror_rate','srv_serror_rate','num_shells','num_root','root_shell','num_compromised','hot','urgent','land','wrong_fragment','count','srv_count'], axis=1, inplace=True)
        #Change Label Contents
        print("File Opened: " +filename1)
        #return test

    #Pre-Processor
    def Process(self):
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()


    # extract numerical attributes and scale it to have zero mean and unit variance  
        cols = self.train.select_dtypes(include=['float64','int64']).columns
        sc_train = scaler.fit_transform(self.train.select_dtypes(include=['float64','int64']))
        sc_test = scaler.fit_transform(self.test.select_dtypes(include=['float64','int64']))

    # turn the result back to a dataframe
        sc_traindf = pd.DataFrame(sc_train, columns = cols)
        sc_testdf = pd.DataFrame(sc_test, columns = cols)

        from sklearn.preprocessing import LabelEncoder
        encoder = LabelEncoder()

        
        # extract categorical attributes from both training and test sets 
        cattrain = self.train.select_dtypes(include=['object']).copy()
        cattest = self.test.select_dtypes(include=['object']).copy()

        # encode the categorical attributes
        traincat = cattrain.apply(encoder.fit_transform)
        testcat = cattest.apply(encoder.fit_transform)

        # separate target column from encoded data 
        enctrain = traincat.drop(['class'], axis=1)
        cat_Ytrain = traincat[['class']].copy()
        
        train_x = pd.concat([sc_traindf,enctrain],axis=1)
        train_y = self.train['class'].replace({'anomaly' : 1, 'normal': 0})
        #print(train_x.shape)
        
        test_df = pd.concat([sc_testdf,testcat],axis=1)
        #print(test_df.shape)
        
        
        from sklearn .ensemble import RandomForestClassifier
        rfc = RandomForestClassifier();

        # fit random forest classifier on the training set
        rfc.fit(train_x, train_y);
        # extract important features
        score = np.round(rfc.feature_importances_,3)
        importances = pd.DataFrame({'feature':train_x.columns,'importance':score})
        importances = importances.sort_values('importance',ascending=False).set_index('feature')
        # plot importances
        plt.rcParams['figure.figsize'] = (11, 4)
        importances.plot.bar();


        from sklearn.feature_selection import RFE
        import itertools
        rfc = RandomForestClassifier()

        # create the RFE model and select 10 attributes
        rfe = RFE(rfc, n_features_to_select=15)
        rfe = rfe.fit(train_x, train_y)

        # summarize the selection of the attributes
        feature_map = [(i, v) for i, v in itertools.zip_longest(rfe.get_support(), train_x.columns)]
        selected_features = [v for i, v in feature_map if i==True]
        

        
        from sklearn.model_selection import train_test_split
        self.X_train,self.X_test,self.Y_train,self.Y_test = train_test_split(train_x,train_y,train_size=0.70, random_state=2)
        
        
        #file = filedialog.asksaveasfilename(initialdir= "F:\Downloads\Knowledge\Programming\Self-Projects\Python Projects\Windowed Neural Network\Data", title="Save X Training Data", filetypes=(("Comma-Seperated Value (*.csv*)", (".csv")),("Text Files (*.txt*)", ".txt"), ("All Files (*.*)", "*.*")), defaultextension="*.*")
        #file1 = filedialog.asksaveasfilename(initialdir= "F:\Downloads\Knowledge\Programming\Self-Projects\Python Projects\Windowed Neural Network\Data", title="Save X Testing Data", filetypes=(("Comma-Seperated Value (*.csv*)", (".csv")),("Text Files (*.txt*)", ".txt"), ("All Files (*.*)", "*.*")), defaultextension="*.*")
        #file2 = filedialog.asksaveasfilename(initialdir= "F:\Downloads\Knowledge\Programming\Self-Projects\Python Projects\Windowed Neural Network\Data", title="Save Y Training Data", filetypes=(("Comma-Seperated Value (*.csv*)", (".csv")),("Text Files (*.txt*)", ".txt"), ("All Files (*.*)", "*.*")), defaultextension="*.*")
        #file3 = filedialog.asksaveasfilename(initialdir= "F:\Downloads\Knowledge\Programming\Self-Projects\Python Projects\Windowed Neural Network\Data", title="Save Y Testing Data", filetypes=(("Comma-Seperated Value (*.csv*)", (".csv")),("Text Files (*.txt*)", ".txt"), ("All Files (*.*)", "*.*")), defaultextension="*.*")

        
        
        #self.X_train.to_csv(file, index = False)
        #self.X_test.to_csv(file1, index = False)
        #self.Y_train.to_csv(file2, index = False)
        #self.Y_test.to_csv(file3, index = False)
        #messagebox.showinfo("Data Shapes","Shape of Test Data is: "+ str(train_x.shape)+ " Shape of Train Data is: "+ str(train_y.shape))
        #X_train = self.X_train.to_csv()
        #X_test = self.X_test.to_csv()
        #Y_train = self.Y_train.to_csv()
        #Y_test = self.Y_test.to_csv()
        
        print("Data Pre-Processed")
        #return X_train, X_test, Y_train, Y_test
        
    
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
        model.fit(self.X_train,
              self.Y_train,
              epochs=5,
              batch_size=10,
              verbose=2)
        return model
    
        #This Section is for the grid search
    def classifyModel(self, epochs):
        #number = int(self.epochs.get())
        neural_network = KerasClassifier(build_fn=self.create_network,
                                         epochs=epochs,
                                         batch_size=10,
                                         #verbose=4
                                         )
        print("The Cross Validation Scores are: ", cross_val_score(neural_network, self.X_train, self.Y_train, cv =3))

    
    
    def gridSearch(self):
        batch_size = [10, 20, 40, 60, 80, 100]
        epochs = [10]
        #Define grid library
        param_grid = dict(batch_size=batch_size, epochs=epochs)
        scorers = {'precision_score': make_scorer(precision_score),
                   'recall_score': make_scorer(recall_score),
                   'accuracy_score': make_scorer(accuracy_score)}
        grid = GridSearchCV(estimator=self.create_network, param_grid=param_grid, n_jobs=-1, cv=3, scoring=scorers,refit="precision_score")
        grid_result = grid.fit(self.X_train, self.Y_train)

        #Preparing for model reports
        y_pred = self.create_network.predict(self.X_test)
        y_pred = (y_pred > 0.5)
        y_pred = y_pred.astype(int)
        cm = confusion_matrix(self.y_test, y_pred)

        print()
        print('============================== Neural Network Confusion Matrix ==============================')
        print(cm)
        print()
    
        print()
        print('============================== Neural Network Accuracy Score ==============================')
        print('Accuracy:', accuracy_score(self.y_test, y_pred))
        print()
    
        print()
        print('============================== Neural Network Classification Report ==============================')
        print(classification_report(self.y_test, y_pred))
        print()
    
        #Remove "#" when grid search is needed
        print()
        print('============================== Neural Network GridSearch ==============================')
        print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
        print()