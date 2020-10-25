import numpy as np
import matplotlib.pyplot as plt

def shuffle_data(data):
    data_shf = {}
    x = data['X']
    t = data['t']
    #set the seed to 1
    np.random.seed(1)
    shuffler = np.random.permutation(len(x))
    x_shuffled = x[shuffler]
    t_shuffled = t[shuffler]
    
    data_shf['X'] = x_shuffled
    data_shf['t'] = t_shuffled
    return data_shf


def split_data(data,num_folds,fold):
    X = data['X']
    t = data['t']
    
    data_fold = {}
    data_rest = {}
    
    #First we are going to calculate each size of part
    block_size = len(t) // num_folds
    s = block_size * (fold-1)
    e = block_size *fold
    
    #get the value of data_fold
    data_fold['X'] = X[s:e,:]
    data_fold['t']= t[s:e]
    
    #get the value of data_rest
    data_rest['X'] = np.concatenate((X[:s,:], X[e:,:]))
    data_rest['t'] = np.concatenate((t[:s],t[e:]))
    
    return data_fold,data_rest  

def train_model(data,lambd):
    X = data['X']
    t = data['t']
    D = len(X[0])
    N = float(len(t))
    
    A = np.dot(X.T,X) + lambd * N * np.identity(D)
    A_inverse = np.linalg.inv(A)
    c = np.dot(X.T,t)
    model = np.dot(A_inverse,c)
    return model

def predict(data,model):
    #return the prediction based on data and model
    X = data['X']
    predictions = np.dot(X,model)
    return predictions

def loss(data,model):
    t = data['t']
    N = float(len(t))
    prediction = predict(data, model)
    square = np.dot((prediction - t).T,(prediction - t))
    error = square/(2*N)
    return error
    
def cross_validation(data,num_folds,lambd_seq):
    cv_error = np.zeros(len(lambd_seq))
    data = shuffle_data(data)
    for i in range(0,len(lambd_seq)):
        lambd = lambd_seq[i]
        cv_loss_lmd = 0
        for fold in range(1,num_folds+1):
            val_cv,train_cv = split_data(data,num_folds,fold)
            model = train_model(train_cv,lambd)
            cv_loss_lmd += loss(val_cv,model)
        cv_error[i] = cv_loss_lmd / num_folds
    return cv_error

def report_error(lambd_seq,data_train,data_test):
    train_set = []
    test_set = []
    for l in lambd_seq:
        model = train_model(data_train,l)
        train_error = loss(data_train,model)
        test_error = loss(data_test,model)
        train_set.append(train_error)
        test_set.append(test_error)
        print("For lambd " + str(l) +": \nThe training errors is "+str(train_error)+\
             ". The test error is "+ str(test_error) + ".")
    return train_set,test_set

def plot(lambd_seq,train,test,five_fold,ten_fold):
    
    #generate a lot showing the errors
    plt.plot(lambd_seq, train,color = 'blue',label='Training error')
    plt.plot(lambd_seq, test,color = 'orange',label='Test error')
    plt.plot(lambd_seq, five_fold,color = 'red',label='5-fold cross validation error')
    plt.plot(lambd_seq, ten_fold,color = 'green',label='10-fold cross validation error')
    
    
    plt.xlabel('lambd')
    plt.ylabel('Errors')
    plt.title('Errors vs Lambda over (0.00005,0.005)')
    plt.legend(loc='best')
    plt.show() 
    
    return;

if __name__ == '__main__':
    data_train = {'X': np.genfromtxt('data/data_train_X.csv',delimiter=','),
                  't': np.genfromtxt('data/data_train_y.csv',delimiter=',')}
    data_test = {'X': np.genfromtxt('data/data_test_X.csv',delimiter=','),
                 't': np.genfromtxt('data/data_test_y.csv',delimiter=',')}
    lambd_seq = np.linspace(0.00005,0.005,num=50)
    train,test = report_error(lambd_seq,data_train,data_test)
    five_fold = cross_validation(data_train,5,lambd_seq)
    ten_fold = cross_validation(data_train,10,lambd_seq)
    #get the lambd with the minimum errors
    lambd_five = lambd_seq[np.argmin(five_fold)]
    lambd_ten = lambd_seq[np.argmin(ten_fold)]
    print("The lambd value with minimum cross validation errors for 5-fold is: "+ \
          str(lambd_five) + "\nThe cross validation errors at this point is :" + \
          str(min(five_fold)))
    print("The lambd value with minimum cross validation errors for 10-fold is: "+ \
          str(lambd_ten) + "\nThe cross validation errors at this point is :" + \
          str(min(ten_fold)))    
    plot(lambd_seq,train,test,five_fold,ten_fold)
