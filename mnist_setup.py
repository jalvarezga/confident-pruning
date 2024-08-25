import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from sklearn.metrics import confusion_matrix
import seaborn as ans
from keras.datasets import mnist
(x_train, y_train), (x_test, y_test)=mnist.load_data()
