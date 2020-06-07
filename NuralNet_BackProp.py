# Name : Amulya Huchachar
# PSU ID : 906527975

# Import all the required Pacakges
import csv
import numpy as np
import random
import matplotlib.pyplot as plt
import math

#global variables
'''
train_data_normalized = np.array([[]])
test_data_nomralized = np.array([[]])
train_lable = np.array({})
test_lable = np.array({})
'''

# function to compute the sigmoid value for the given input
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# function to compute the sigmoid derivation.
def sigmoid_derivative(sigmoid_output):
    return sigmoid_output * (1-sigmoid_output)

# function to calculate the dot product and apply sigmoid funtion on the result.
def activate(inputs, weights):
    return sigmoid(np.dot(inputs, weights.transpose()))


# creating a class for back propagation algorithm  
class BackPropagate:

    ''' __init__ function : Used to initialize the data.
                self : referece to the calling instance of the class object
                n_inputs : No of input units
                n_hidden : No of hidden units
                n_output : No of output units
                learning_rate : to set the learning rate
    '''

    def __init__(self, n_inputs, n_hidden, n_output, learning_rate):
        self.n_inputs = n_inputs
        self.n_hidden = n_hidden
        self.n_output = n_output

        self.learning_rate = learning_rate

        # initializing the input and hidden unit weights randomely  
        self.input_to_hidden_weight = np.random.uniform(-0.05,0.05,(n_hidden,n_inputs+1))
        self.hidden_to_output_weight = np.random.uniform(-0.05, 0.05,(n_output, n_hidden+1))

        # Creaing the bias unit for input and hidden layer
        self.input_to_hidden_bias = np.ones((n_hidden,1))
        self.hidden_to_output_bias = np.ones((n_output,1))
   
        self.previous_outputWeightDelta = np.array([])
        self.previous_hiddenWeightDelta = np.array([])
        self.outputWeightDelta = np.array([])
        self.hiddenWeightDelta = np.array([])
        self.momentum = 0


    '''
        train : Function to traing the model
        parametrts : 
                self : referece to the calling instance of the class object
                epoc : number of epocs 
    '''
    def train(self, epoc, train_data_normalized, train_lable):
        self.totalError = 0
        # for all the images
        for i in range(30000):
            #forward propogate to predict the output
            self.forwardPropogation(train_data_normalized[i,:])
            #compute error of the prediction
            self.error(train_lable[i,0])
            #update global accuracy
            self.updateAccurecy(i,train_lable[i,0])
            #backward propogate to update the weights
            if(epoc != 0):
                self.backPropogation()
        #total error        
        self.totalError = self.totalError / train_data_normalized.shape[0]     
        # return total accuracy 
        return (1 - (self.totalError)) * 100

    '''
        test : Function to test the model
        Parameters :
                 self : referece to the calling instance of the class object
    '''     
    def test(self, test_data_normalized, test_lable): 
        #total error
        self.totalError = 0
        # confusion matrix
        self.confusionMatrix = np.zeros((10,10), dtype=int)
        # for all the test images
        for i in range(test_data_normalized.shape[0]):
            # forward propogate to predict the output 
            self.forwardPropogation(test_data_normalized[i,:])     
            #compute error of the prediction  
            self.error(test_lable[i,0])  
            #update confusion matrix
            self.confusionMatrix[test_lable[i,0] , np.argmax(self.output)] = (self.confusionMatrix[test_lable[i,0] , np.argmax(self.output)]) + 1
            #update global accuracy
            self.updateAccurecy(i,test_lable[i,0])

         #total error  
        self.totalError = self.totalError / test_data_normalized.shape[0]  
        print(self.confusionMatrix)   
        # return total accuracy      
        return (1 - (self.totalError)) * 100

    '''
        forwardPropogation : Function to feed forward a input data to pridect label
        Parameters : 
                self : referece to the calling instance of the class object
                image : pixle details of a input image.
    '''
    def forwardPropogation(self, image):
        #append bais to the input data
        self.inputWithBias = image.reshape(1,785)
        #activate neurons in the first layer
        self.hidden = activate(self.inputWithBias, self.input_to_hidden_weight)
        #append bias to the hidden layer
        self.hiddenWithBias = np.insert(self.hidden, 0, 1., axis=1)
        #activate neurons in the hidden layer
        self.output = activate(self.hiddenWithBias, self.hidden_to_output_weight)  
        
    '''
        error : Function to compute the error for the prideciton 
        Parameters :
                self : referece to the calling instance of the class object
                lable : currect label
    '''
    def error(self, label):
        #create one hot encoding 
        lable_one_hot = np.zeros((1,10))
        #update one hot encoding
        lable_one_hot[:] = 0.1
        lable_one_hot[0,label] = 0.9
        # determin error from output layer
        self.errorNonSigmoid = (lable_one_hot- self.output)
        # apply sigmoid derivative
        self.outputError = sigmoid_derivative(self.output) * (lable_one_hot- self.output)
        # determin error from hidden layer
        self.hiddenError = sigmoid_derivative(self.hidden) * np.dot(self.outputError, self.hidden_to_output_weight[:,1:])

    '''
        backPropogation : Function to perform the back propagation after finding the error factor
        Parameters :
                self : referece to the calling instance of the class object
    '''
    def backPropogation(self):
        # store previous weight delta for moment update
        self.previous_outputWeightDelta = self.outputWeightDelta 
        self.previous_hiddenWeightDelta = self.hiddenWeightDelta
        
        # compute current weight delta 
        self.outputWeightDelta = self.learning_rate * np.dot(self.hiddenWithBias.transpose(), self.outputError)
        self.hiddenWeightDelta = self.learning_rate * np.dot(self.inputWithBias.transpose(), self.hiddenError)
        
        # add the momentum factor to the weight delta
        if  self.previous_outputWeightDelta.size != 0 and self.momentum !=0:
            self.outputWeightDelta = self.outputWeightDelta + (self.momentum * self.previous_outputWeightDelta)
            self.hiddenWeightDelta = self.hiddenWeightDelta + (self.momentum *self.previous_hiddenWeightDelta)
      
        # update weigths 
        self.hidden_to_output_weight = self.hidden_to_output_weight + self.outputWeightDelta.transpose()
        self.input_to_hidden_weight = self.input_to_hidden_weight + self.hiddenWeightDelta.transpose()

    '''
        updateAccurecy : Function to determine the accuracy of the model
    '''
    def updateAccurecy(self, i, label):
        if(np.argmax(self.output) != label ):
            self.totalError += 1
    '''
        store_accur : Function to write data to text file for ploting
    '''
    def store_accur(self, accur_index,accur,input_ds):
        with open(input_ds, 'a', newline='') as myfile:
            wr = csv.writer(myfile)
            wr.writerow([accur_index,accur])

def prepareData():
    # Store the MNIST dataset CSV file names to load data   
    training_data = 'mnist_train.csv'
    testing_data = 'mnist_test.csv'
    
    # Load mnist training dataset
    f = open(training_data,'r')
    data = csv.reader(f)
    list_data = list(data)
    train_data = np.array(list_data)

    # Load  mnist test dataset
    f1 = open(testing_data,'r')
    data1 = csv.reader(f1)
    list_data1 = list(data1)
    test_data = np.array(list_data1)

    # Separate the image lables and image pixel data for both training and test data
    # normalize data
    fac = 0.99/255
    train_data_normalized = np.asfarray(train_data[:,0:]) * fac + 0.01
    train_data_normalized[:,0] = 1
    test_data_normalized = np.asfarray(test_data[:,0:]) * fac + 0.01
    test_data_normalized[:,0] =1
    train_lable = np.asarray(train_data[:,:1], dtype=int)
    test_lable = np.asarray(test_data[:,:1], dtype=int)

    return train_data_normalized, train_lable, test_data_normalized, test_lable

'''
 uitlity function to plot the test and train accuracies 
'''
def plot(trainFileName, testFineName, title):
    x1, y1 = np.loadtxt(trainFileName,delimiter=',',unpack=True)
    x2, y2 = np.loadtxt(testFineName,delimiter=',',unpack=True)
    plt.plot(x1,y1, label="Training Set")
    plt.plot(x2,y2, label="Testing Set")
    plt.xlabel('Epochs')
    plt.title(title)
    plt.ylabel('Accuracy (%) ')
    plt.legend(loc='lower right')
    plt.show()

def main():
    #create object to train and test a backPropagation model
    backPropagate = BackPropagate(784,100,10,0.1)

    #prepare data
    train_data_normalized, train_lable, test_data_normalized, test_lable = prepareData()
   
    #file to store accuracies for ploting
    trainFileName = 'train_backProp_30k_output'
    testFineName = 'test_backProp_30k_demo.csv'
    title = 'For Module trained with 30000 records '
    
    #number of epocs
    numEpocs = 50
    #for each epoc
    for epoc in range(numEpocs):
        #train the model
        trainAccuracy = backPropagate.train(epoc,train_data_normalized, train_lable)
        #test the model
        testAccuracy = backPropagate.test(test_data_normalized, test_lable)
        print("Epoc " + str(epoc) + " => TrainAccuracy : " +str(trainAccuracy) +" , TestAccracy :  "+str(testAccuracy))  
    
        #store train and test accurancy in a file for ploting
        backPropagate.store_accur(epoc,trainAccuracy,trainFileName)
        backPropagate.store_accur(epoc,testAccuracy,testFineName)

    #plot the accuracies
    plot(trainFileName, testFineName, title)    
    

if __name__== "__main__":
  main()



