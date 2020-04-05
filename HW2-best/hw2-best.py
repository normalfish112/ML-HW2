import numpy as np
import argparse 

parser = argparse.ArgumentParser()
parser.add_argument('pretrain', type=str, help="X_train")
parser.add_argument('pretest', type=str, help="X_test")
parser.add_argument('output', type=str, help="output path")

args = parser.parse_args()
X_train_fpath = args.pretrain
X_test_fpath = args.pretest
output_fpath = args.output

# Parse csv files to numpy array
with open(X_train_fpath) as f:
    next(f)
    X_train = np.array([line.strip('\n').split(',')[1:] for line in f], dtype = float)
with open(X_test_fpath) as f:
    next(f)
    X_test = np.array([line.strip('\n').split(',')[1:] for line in f], dtype = float)

mask1,mask2 = [],[508,509]
for i in range(351-323+1):
    mask2 += [323+i-1]
mask = mask1+mask2
X_train = np.delete(X_train,mask,1)
X_test = np.delete(X_test,mask,1)

X_train =  np.concatenate((X_train,X_train**2,X_train**3),axis=1)
X_test =  np.concatenate((X_test, X_test**2, X_test**3),axis=1)

def _normalize(X, train = True, specified_column = None, X_mean = None, X_std = None):
    if specified_column == None:
        specified_column = np.arange(X.shape[1])
    if train:
        X_mean = np.mean(X[:, specified_column] ,0).reshape(1, -1)
        X_std  = np.std(X[:, specified_column], 0).reshape(1, -1)

    X[:,specified_column] = (X[:, specified_column] - X_mean) / (X_std + 1e-8)
     
    return X, X_mean, X_std

# Normalize training and testing data
X_train, X_mean, X_std = _normalize(X_train, train = True)
X_test, _, _= _normalize(X_test, train = False, specified_column = None, X_mean = X_mean, X_std = X_std)
    
########################################
#Some Useful Functions
def _sigmoid(z):
    # Sigmoid function can be used to calculate probability.
    # To avoid overflow, minimum/maximum output value is set.
    return np.clip(1 / (1.0 + np.exp(-z)), 1e-8, 1 - (1e-8))

def _f(X, w, b):
    # This is the logistic regression function, parameterized by w and b
    #
    # Arguements:
    #     X: input data, shape = [batch_size, data_dimension]
    #     w: weight vector, shape = [data_dimension, ]
    #     b: bias, scalar
    # Output:
    #     predicted probability of each row of X being positively labeled, shape = [batch_size, ]
    return _sigmoid(np.matmul(X, w) + b)

def _predict(X, w, b):
    # This function returns a truth value prediction for each row of X 
    # by rounding the result of logistic regression function.
    return np.round(_f(X, w, b)).astype(np.int)

########################################
#Plotting Loss and accuracy curve
# Predict testing labels
w = np.load('best_weight.npy')  
b = np.load('best_bias.npy') 
predictions = _predict(X_test, w, b)
with open(output_fpath, 'w') as f:
    f.write('id,label\n')
    for i, label in  enumerate(predictions):
        f.write('{},{}\n'.format(i, label))