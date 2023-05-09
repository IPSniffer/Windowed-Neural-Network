# import relevant modules
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
#import seaborn as sns
import sklearn as sns
#import imblearn

# Ignore warnings
import warnings
warnings.filterwarnings('ignore')

#Data assigned variables and attributes removed
train = pd.read_csv("Train_data.csv")
test = pd.read_csv("Test_data.csv")
train.drop(['num_outbound_cmds','dst_host_srv_rerror_rate','dst_host_rerror_rate','dst_host_srv_serror_rate','dst_host_serror_rate','dst_host_srv_diff_host_rate','dst_host_same_src_port_rate','dst_host_diff_srv_rate','dst_host_same_srv_rate','dst_host_srv_count','dst_host_count','srv_diff_host_rate','diff_srv_rate','same_srv_rate','srv_rerror_rate','rerror_rate','srv_serror_rate','num_shells','num_root','root_shell','num_compromised','hot','urgent','land','wrong_fragment','count','srv_count'], axis=1, inplace=True)
test.drop(['num_outbound_cmds','dst_host_srv_rerror_rate','dst_host_rerror_rate','dst_host_srv_serror_rate','dst_host_serror_rate','dst_host_srv_diff_host_rate','dst_host_same_src_port_rate','dst_host_diff_srv_rate','dst_host_same_srv_rate','dst_host_srv_count','dst_host_count','srv_diff_host_rate','diff_srv_rate','same_srv_rate','srv_rerror_rate','rerror_rate','srv_serror_rate','num_shells','num_root','root_shell','num_compromised','hot','urgent','land','wrong_fragment','count','srv_count'], axis=1, inplace=True)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

# extract numerical attributes and scale it to have zero mean and unit variance  
cols = train.select_dtypes(include=['float64','int64']).columns
sc_train = scaler.fit_transform(train.select_dtypes(include=['float64','int64']))
sc_test = scaler.fit_transform(test.select_dtypes(include=['float64','int64']))

# turn the result back to a dataframe
sc_traindf = pd.DataFrame(sc_train, columns = cols)
sc_testdf = pd.DataFrame(sc_test, columns = cols)

from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()


# extract categorical attributes from both training and test sets 
cattrain = train.select_dtypes(include=['object']).copy()
cattest = test.select_dtypes(include=['object']).copy()

# encode the categorical attributes
traincat = cattrain.apply(encoder.fit_transform)
testcat = cattest.apply(encoder.fit_transform)

# separate target column from encoded data 
enctrain = traincat.drop(['class'], axis=1)
cat_Ytrain = traincat[['class']].copy()

train_x = pd.concat([sc_traindf,enctrain],axis=1)
train_y = train['class']
print(train_x.shape)

test_df = pd.concat([sc_testdf,testcat],axis=1)
print(test_df.shape)

from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()

# extract categorical attributes from both training and test sets 
cattrain = train.select_dtypes(include=['object']).copy()
cattest = test.select_dtypes(include=['object']).copy()

# encode the categorical attributes
traincat = cattrain.apply(encoder.fit_transform)
testcat = cattest.apply(encoder.fit_transform)

# separate target column from encoded data 
enctrain = traincat.drop(['class'], axis=1)
cat_Ytrain = traincat[['class']].copy()

train_x = pd.concat([sc_traindf,enctrain],axis=1)
train_y = train['class']
print(train_x.shape)

test_df = pd.concat([sc_testdf,testcat],axis=1)
print(test_df.shape)

from sklearn.ensemble import RandomForestClassifier
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

#print(selected_features)

from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(train_x,train_y,train_size=0.70, random_state=2)

#Export pre-processed data into csv file
X_train.to_csv(r"C:\Users\IPSniffer\Documents\Work\University\Year 3\Comp Sci Project\Project Files\Pre-Processed_X_Train_data.csv", index = False)
Y_train.to_csv(r"C:\Users\IPSniffer\Documents\Work\University\Year 3\Comp Sci Project\Project Files\Pre-Processed_Y_Train_data.csv", index = False)
X_test.to_csv(r"C:\Users\IPSniffer\Documents\Work\University\Year 3\Comp Sci Project\Project Files\Pre-Processed_X_Test_data.csv", index = False)
Y_test.to_csv(r"C:\Users\IPSniffer\Documents\Work\University\Year 3\Comp Sci Project\Project Files\Pre-Processed_Y_Test_data.csv", index = False)

