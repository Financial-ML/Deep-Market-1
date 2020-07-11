#Dependencies
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import SGD
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from tensorflow import keras
#------------------------------------------------------------------------------
# Initialize the MLP
def initialize_nn(frame_size):
    model = Sequential() # The Keras Sequential model is a linear stack of layers.
    model.add(Dense(800, init='uniform', input_dim=frame_size)) # Dense layer
    model.add(Activation('tanh')) # Activation layer
    model.add(Dropout(0.2)) # Dropout layer
    model.add(Dense(800, init='uniform')) # Another dense layer
    model.add(Activation('tanh')) # Another activation layer
    model.add(Dropout(0.2)) # Another dropout layer
    model.add(Dense(2, init='uniform')) # Last dense layer
    model.add(Activation('softmax')) # Softmax activation at the end
    opt = keras.optimizers.Adam(learning_rate=0.00001)
    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy']) # Using logloss
    return model
#------------------------------------------------------------------------------
def generate_results(y_test, y_score):
    fpr, tpr, _ = roc_curve(y_test, y_score)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.05])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic curve')
    plt.show()
    print('AUC: %f' % roc_auc)
#------------------------------------------------------------------------------   
#dataset import
dataset = pd.read_csv('calculated.csv') #You need to change #directory accordingly
#------------------------------------------------------------------------------
#Changing pandas dataframe to numpy array
X = dataset.iloc[:,:71].values
y = dataset.iloc[:,66:67].values
X = np.delete(X, np.s_[::66], 1)
X = X.astype(np.float64)
n_frames, frame_size = X.shape
#------------------------------------------------------------------------------
from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder(categories='auto')
y = ohe.fit_transform(y).toarray()
#------------------------------------------------------------------------------
print('Splitting data into training and testing')
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)
#------------------------------------------------------------------------------
print('Initializing model')
model = initialize_nn(frame_size)
#------------------------------------------------------------------------------
history = model.fit(X_train, y_train,validation_data = (X_test,y_test), epochs=250, batch_size=100)
#------------------------------------------------------------------------------
y_pred = model.predict(X_test)
#Converting predictions to label
pred = list()
for i in range(len(y_pred)):
    pred.append(np.argmax(y_pred[i]))
#Converting one hot encoded test label to label
test = list()
for i in range(len(y_test)):
    test.append(np.argmax(y_test[i]))
#------------------------------------------------------------------------------
from sklearn.metrics import accuracy_score
a = accuracy_score(pred,test)
print(" ")
print('Accuracy is:', a*100)
#------------------------------------------------------------------------------
#save model
# serialize model to YAML
model_yaml = model.to_yaml()
with open("model.yaml", "w") as yaml_file:
    yaml_file.write(model_yaml)
# serialize weights to HDF5
model.save_weights("model.h5")
print("Saved model to disk")
#------------------------------------------------------------------------------
 