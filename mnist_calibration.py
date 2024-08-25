
from scipy.stats import binom #to obtain the  valid-pvalue
# Import math Library
import math# to use functions such as log
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
print(tf.version.VERSION)
import keras
keras.version()
from keras.models import Sequential
from keras.layers import Dense, Dropout
from sklearn.metrics import confusion_matrix
import seaborn as ans
np.random.seed(0) #for reproducibility
from keras.datasets import mnist
(x_train, y_train), (x_test_original, y_test_original)=mnist.load_data()
#we actually do not need the training dataset. We only incroporate it to have the original split 
#such that the train data is kept separate from all the calibration process
#x_train=x_train/255.0 #normalize so that all the entries are in [0,1]
x_test_original=x_test_original/255.0
num_classes=10
y_test_original=keras.utils.to_categorical(y_test_original, num_classes)# convert one-hot-encoding format
#x_train=x_train.reshape(x_train.shape[0], -1)# to convert to one dimension---  aka long vectors
x_test_original=x_test_original.reshape(x_test_original.shape[0], -1)
# Generate random sample
#np.random.seed(0) #for reproducibility
vector_range = np.arange(10000)  # Create array from 0 to 9999
sample_size = 1000  # Number of samples to draw
sample = np.random.choice(vector_range, size=sample_size, replace=False)
print(len(sample))
#print(sample)
x_validate=x_test_original[sample]
y_validate=y_test_original[sample]
x_calibrate=x_test_original[np.setdiff1d(vector_range, sample)]
y_calibrate=y_test_original[np.setdiff1d(vector_range, sample)]
print(x_test_original.shape, 'train images')#now they are 784 rowed columns and they are 10,000!!
print(x_calibrate.shape, 'calibration images') #9,000 images
print(x_validate.shape, 'coverage-validation images') #1,000 images
lambdas_string = [f"{0.01 * i:.2f}" for i in range(100)] #string representation of the pruning ratios.
#print(lambdas_string) #this is useful so that when we load the models, there's no issue at 
lambdas=[i/100 for i in range(100)]
models={} #to store all the models in a dictionary in a key:value format. Key is the value of lambda as a string, value is the model
#make sure to use your working directory where you saved the pruned neural networks!
for i in lambdas_string:
        best_model_file = f'/Users/joaquinalvarez/Downloads/pruning2/pruned_model_{i}.keras'
        model= tf.keras.models.load_model(best_model_file)
        models[i]=model
vector_range = np.arange(10000)  # Create array from 0 to 9999
def empirical_risks(seed_argument):
    #this function will provide the empirical risks obteined accross all the pruned models, including the original model
    #and it will provide the risks with the calibration dataset and the test dataset
    np.random.seed(seed_argument) #for reproducibility
    sample = np.random.choice(vector_range, size=sample_size, replace=False)
    x_validate=x_test_original[sample]#we use 1,000 images to evaluate coverage 
    y_validate=y_test_original[sample]
    x_calibrate=x_test_original[np.setdiff1d(vector_range, sample)] #we use 9,000 images  to test/ calibrate
    y_calibrate=y_test_original[np.setdiff1d(vector_range, sample)]
    empirical_risks_validation=[]
    empirical_risks_calibration=[]
    for i in lambdas_string:
        validate_loss, validate_acc=models[i].evaluate(x_validate, y_validate)
        empirical_risks_validation.append(1-validate_acc)
        calibrate_loss, calibrate_acc=models[i].evaluate(x_calibrate, y_calibrate)
        empirical_risks_calibration.append(1-calibrate_acc)
    return({'validation':empirical_risks_validation, 'calibration':empirical_risks_calibration})
#print(empirical_risks(1))
#valid p-value function
def valid_p_val(r,alpha,n):
    return(binom.cdf(k=r*n, n=n, p=alpha))
simulation_example=empirical_risks(1)
#print(simulation_example)
def fixed_sequence(delta,alpha,n, empirical_risks):
    #delta is the level at which  we wish to implement the fixed sequence testing
    #alpha, the risk control threshold
    #n the size of the sample 
    k=0#initialize the index of the first null hypothesis to be examined 
    while  valid_p_val(empirical_risks[k],alpha, n)<delta:
        k=k+1 #we reject the null and explore pruning more parameters to see if we should reject the null or not
    return(lambdas[k-1],k-1) #also useful to have the corresponding index to map between valid p_values
#why return k-1 and not k? becasue  we exit the loop at valid_p_val(risks[k],alpha, n)>=delta. That is, a k such that the associated null was NOT rejected
#and empirircal risks and pruning parameter indicdes in lambdas

#implementation example
print(len(x_calibrate)) #prints 9,000, total number of images in the calibration dataset

print(fixed_sequence(delta=0.05, alpha=0.03, n=len(x_calibrate),empirical_risks=simulation_example['calibration']), 'was the output for delta=0.05, alpha=0.03' )
