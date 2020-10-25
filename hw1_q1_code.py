import sklearn
import matplotlib

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

import matplotlib.pyplot as plt

def train_val_test_split(dataX,dataY):
    
    # let train data be the 70% of the entire data_set
    x_train, x_test_val, y_train, y_test_val = \
    train_test_split(dataX, dataY, test_size = 0.3, random_state=10)

    # let test and validation data be the 15% of the entire data_set
    x_validation, x_test, y_validation, y_test = \
        train_test_split(x_test_val, y_test_val, test_size = 0.5, random_state=10) 

    return x_train, x_validation, x_test, y_train, y_validation, y_test

def load_data():
    # X
    headlines = []
    # Y
    labels = []
    # add the fake headlines to list
    with open('data/clean_fake.txt', 'r') as file:
        #interate over the headlines in fake file
        for line in file:
            headlines.append(line.strip('\n'))
    fake_len = len(headlines)
    
    for i in range(0,fake_len):
        labels.append(0)
    
    # add the real headlines to list
    with open('data/clean_real.txt', 'r') as file:
        #interate over the headlines in fake file
        for line in file:
            headlines.append(line.strip('\n'))    
    real_len = len(headlines) - fake_len      
    
    for i in range(0,real_len):
        labels.append(1)
        
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(headlines)
    y = labels
    return train_val_test_split(X,y)

def plot(k_set,accur_train,accur_valid,i):
    
    #generate a lot showing the accuracy for both training and validation
    plt.plot(k_set, accur_train,color = 'blue',label='Training accuracy')
    plt.plot(k_set, accur_valid,color = 'orange',label='Validation accuracy')
    
    plt.xlabel('k')
    plt.ylabel('Accuracy')
    plt.title('Training/Validation set accuracy v.s. k')
    plt.figure(i)
    plt.legend(loc='best')
    plt.show() 
    
    return;
      
def select_knn_model(x_train,y_train,x_valid,y_valid):
    best_model = None
    best_accur = 0
    k_set = []
    accur_train = []
    accur_valid = []
    
    for k in range(1,21): 
        neigh = KNeighborsClassifier(n_neighbors=k)
        #fit the model on data
        neigh.fit(x_train,y_train)
        
        #calculate the training accuracy
        training_accur= neigh.score(x_train,y_train)
        #calculate the training errors
        training_error = 1-training_accur
        #calculate the validation accuracy
        validation_accur = neigh.score(x_valid,y_valid)
        #calculate the validation errors
        validation_error = 1-validation_accur        
        
        k_set.append(k)
        #add the accuracy to the list respectly
        accur_train.append(training_accur)
        accur_valid.append(validation_accur)
          
        #check if it's best model so far
        if(validation_accur > best_accur):
            best_model = neigh
            best_accur = validation_accur
    
    plot(k_set,accur_train,accur_valid,0)
    return best_model,best_accur

def select_knn_model_cosine(x_train,y_train,x_valid,y_valid):
    best_model = None
    best_accur = 0
    k_set = []
    accur_train = []
    accur_valid = []
    
    for k in range(1,21): 
        neigh = KNeighborsClassifier(n_neighbors=k,metric='cosine')
        #fit the model on data
        neigh.fit(x_train,y_train)
        
        #calculate the training errors
        training_accur= neigh.score(x_train,y_train)
        validation_accur = neigh.score(x_valid,y_valid)
        
        k_set.append(k)
        #add the accuracy to the list respectly
        accur_train.append(training_accur)
        accur_valid.append(validation_accur)
          
        #check if it's best model so far
        if(validation_accur > best_accur):
            best_model = neigh
            best_accur = validation_accur
    
    plot(k_set,accur_train,accur_valid,1)
    return best_model,best_accur

if __name__ == '__main__':
    x_train, x_valid, x_test, y_train, y_valid, y_test = load_data()
    #get the model witg best validation accuracy at default case
    best_model,best_accur = select_knn_model(x_train,y_train,x_valid,y_valid)
    print("The model with best validation accuracy: " + str(best_model))
    test_accur = best_model.score(x_test,y_test)
    print("The accuracy on the test data is: " + str(test_accur))
    #get the model witg best validation accuracy when metric is cosine
    best_model,best_accur = select_knn_model_cosine(x_train,y_train,x_valid,y_valid)
    print("The model with best validation accuracy: " + str(best_model))
    test_accur = best_model.score(x_test,y_test)
    print("The accuracy on the test data is: " + str(test_accur))    