import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from sklearn.metrics import confusion_matrix
import seaborn as ans
from keras.datasets import mnist
(x_train, y_train), (x_test, y_test)=mnist.load_data()
np.random.seed(0) #for reproducibility
#Visualize some examples of the MNIST dataset
num_classes=10
f,ax =plt.subplots(1, num_classes, figsize=(20,20))
for i in range(0, num_classes):
  sample=x_train[y_train==i][0] #take an image (the first one) whose
  #corresponding number is equal to i
  ax[i].imshow(sample, cmap='gray')# show the image in gray scale format (black and white allowing intensities)
  ax[i].set_title(f'Label {i}', fontsize=20)

#normalize the data
x_train=x_train/255.0
x_test=x_test/255.0
x_train=x_train.reshape(x_train.shape[0], -1)# to convert to one dimension
x_test=x_test.reshape(x_test.shape[0], -1)
#np.random.seed(0) #for reproducibility
vector_range = np.arange(10000)  # Create array from 0 to 9999
sample_size = 1000  # Number of samples to draw
sample = np.random.choice(vector_range, size=sample_size, replace=False)

#train a neural network
model=Sequential()
model.add(Dense(units=128, input_shape=(784,), activation='relu')) #add an input layer
model.add(Dense(units=128, activation='relu')) #hidden layer
model.add(Dropout(.25)) #apply dropout to avoid overfitting during training
model.add(Dense(units=10, activation='softmax')) #to obtain outputs in [0,1] use a suft-max activation function
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()
batch_size=512
epochs=10
model.fit(x=x_train, y=y_train, batch_size=batch_size, epochs=epochs)
#evaluate the performance of the model 
test_loss, test_acc=model.evaluate(x_test, y_test)
print(f'Test Loss:{test_loss}')
print(f'Test accuracy:{test_acc}')
#save the model
save_model(model, 'model.keras')

# A function to create a pruned model and visualize a heatmap  of the weights
#lamb should be a pruning ratio e.g. lamb=0.72, lamb=0.10 etc.
#make sure to use your own working directory where you saved your model.keras. 
def weight_heatmap(lamb):
    #load the model that we obtained after training
    best_model_file = '/Users/joaquinalvarez/Downloads/pruning2/model.keras'
    model= tf.keras.models.load_model(best_model_file)
    #pruned_model = copy.deepcopy(model)  # Create a copy of the model
    #we extrract all the weights of the model in a flatten list
    weights=model.get_weights()[0].flatten()
    for i in range(1,len(model.get_weights())):
        weights=np.concatenate((weights,model.get_weights()[i].flatten()))
        #print(i) #for debugging and see how we doing

    #Obtain a quantile of the smallest prune_percentage weights in absolute value
    threshold = np.quantile(np.abs(weights),lamb)  # Modify weights whose absolute value is below this threshold

# Iterate through all trainable variables in the copied_model
    for variable in model.trainable_variables:
        #if variable.dtype.is_floating:
        #Modify weights below the threshold
        mask = tf.cast(tf.abs(variable) >= threshold, dtype=variable.dtype)
        new_value = variable * mask  # Zero out weights below the threshold

        # Assign the modified value back to the variable. This means redefining the weights to the new ones
        #after pruning the 'small' weights 
        variable.assign(new_value)

    new_weights=model.get_weights()[0].flatten()
    for i in range(1,len(model.get_weights())):
        new_weights=np.concatenate((new_weights, model.get_weights()[i].flatten()))
        #print(i) #for debugging and see how we doing
    save_model(model, f'pruned_model_{lamb}.keras')
    print('pruned model is saved', f'new pruned model has {sum(new_weights == 0)/new_weights.shape[0]} of the weights equal to zero')
    test_loss, test_acc= model.evaluate(x_test, y_test)
    #print(f'Test Loss:{test_loss}')
    #print(f'Test accuracy:{test_acc}')
    # Assuming model.get_weights()[0] returns the weights array
    weights = model.get_weights()[0]
    # Calculate absolute values of the weights
    abs_weights = np.abs(weights)
    # Plot heatmap
    plt.figure(figsize=(8,10))  # Adjust the figure size as necessary
    #plt.imshow(abs_weights, cmap='hot', interpolation='nearest')
    plt.imshow(abs_weights, cmap='hot', interpolation='nearest',  vmin=0.0, vmax=.25)
    plt.colorbar()  # Add colorbar to show the scale
    np.set_printoptions(precision=3) #to print only 3 decimals
    plt.title(fr'$\lambda=${lamb}, Test accuracy:{test_acc:.3f}')
    plt.xlabel('Neuron in the first hidden layer', fontsize=10)
    plt.ylabel('Input neuron (pixel)')
    # Adjust layout
    #plt.tight_layout()
    plt.savefig(f'pruned_model_{lamb}.png')   #save the fugure if you want to 
    plt.show()


#define a function that generates a  matrix with the average magnitude of the weights connecting the fisrt hidden layer to the input layer
def matrixRepresentation_average_weights(weights_first_layer):
    matr=np.empty((28, 28))
    k=0#initialize a number to move across the weights and average them
    #n we want ton abtain a matrix that computes the average weight that feedst tghe image into the first hidden 
    #layer
    for i in range(28):
        for j in range(28):
            matr[i,j]=np.mean(weights_first_layer[k,:])
            k+=1
    return(matr)
#we use matrixRepresentation_average_weights to 
def visualization_pixels_weights(lamb):
    best_model_file = f'/Users/joaquinalvarez/Downloads/pruning2/pruned_model_{lamb}.keras'
    model= tf.keras.models.load_model(best_model_file)
    #test_loss, test_acc=model.evaluate(x_test, y_test)
    #print(f'Test Loss:{test_loss}')
    #print(f'Test accuracy:{test_acc}') #there's no need to do a complile()!  #we see that after pruning at 50%
    #the accuracy  is still 50%

    # Calculate absolute values of the weights
    abs_weights = np.abs(matrixRepresentation_average_weights(model.get_weights()[0]))

    # Plot heatmap
    plt.figure(figsize=(8, 6))  # Adjust the figure size as necessary
    plt.imshow(abs_weights, cmap='hot', interpolation='nearest',  vmin=0, vmax=.02)
    plt.colorbar().set_label('Average weight magnitude', fontsize=15)  # Add colorbar to show the scale
    plt.title(fr'$\lambda=${lamb}',  fontsize=20)
    #plt.title('Heatmap of mean absolute value of the weights feeding the first hidden layer per pixel')
    plt.xlabel('Input neuron x axis',fontsize=20)
    plt.ylabel('Input neuron y axis',fontsize=20)
    #plt.savefig(f'pruned_model_pixels{lamb}.png') 
    plt.show()
