import numpy as np
import os 
import cv2
import pickle
import copy

#Input Layer
class Layer_Input():

    def forward(self, inputs, training):
        self.output = inputs

class Layer_Dense:
    def __init__(self, n_inputs, n_neurons, weight_regularizer_l1=0, weight_regularizer_l2=0,bias_regularizer_l1=0, bias_regularizer_l2=0):
        #Intialize weights and biases
        self.weights = 0.01*np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))

        #Set regularization strength
        self.weight_regularizer_l1  = weight_regularizer_l1
        self.weight_regularizer_l2  = weight_regularizer_l2
        self.bias_regularizer_l1    = bias_regularizer_l1
        self.bias_regularizer_l2    = bias_regularizer_l2

    def forward(self, inputs, training):
        #Remeber input values
        self.inputs = inputs
        #Calculate output values from inputs, weights and biases
        self.output = np.dot(inputs,self.weights) + self.biases
       

    def backward(self, dvalues):
        #Gradients of params
        self.dweights = np.dot(self.inputs.T, dvalues)
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)

        #Gradient on regularization
        #L1 on weights
        if self.weight_regularizer_l1 > 0:
            dL1 = np.ones_like(self.weights)
            dL1[self.weights < 0] = -1
            self.dweights += self.weight_regularizer_l1 * dL1 
        
        #L2 on weights
        if self.weight_regularizer_l2 > 0:
            self.dweights += 2 * self.weight_regularizer_l2 * self.weights

        #L1 on bias
        if self.bias_regularizer_l1 > 0:
            dL1 = np.ones_like(self.biases)
            dL1[self.bias < 0] = -1
            self.dbiases += self.bias_regularizer_l1 * dL1 
        
        #L2 on bias
        if self.bias_regularizer_l2 > 0:
            self.dbiases += 2 * self.bias_regularizer_l2 * self.biases
           
        #Gradient on values
        self.dinputs = np.dot(dvalues, self.weights.T)

    #Retrieve layer parameters
    def get_parameters(self):
        return self.weights, self.biases

    #Sets weights and biases in a layer instance
    def set_parameters(self, weights, biases):
        self.weights = weights
        self.biases = biases

class Layer_Dropout():
    #Init
    def __init__(self, rate):

        #Store rate, we invert it as for example for dropout of 0.1, we need scuess rate of 0.9
        self.rate = 1 -rate

    #Forward passs
    def forward(self, inputs, training):

        #Save input values
        self.inputs = inputs

        #If not in the training mode - return values:
        if not training:
            self.output = inputs.copy()
            return

        #Generate and save scaled mask
        self.binary_mask = np.random.binomial(1, self.rate, size=inputs.shape) / self.rate

        #Apply mask to output values
        self.output = inputs * self.binary_mask

    def backward(self, dvalues):

        #Gradients on values
        self.dinputs = dvalues * self.binary_mask

class Activation_ReLU:
    def forward(self, inputs, training):
        #Rember input values
        self.inputs = inputs
        #Calculate output values from input. Formula: x if x > 0 else 0
        self.output = np.maximum(0, inputs)
    
    def backward(self, dvalues):
        #Hard copy so that original variables arent modified
        self.dinputs = dvalues.copy()
        
        # Zero grad where input values were neg
        self.dinputs[self.inputs <= 0] = 0

class Activation_Softmax:
    def forward(self, inputs, training):  
        #Store input values
        self.inputs = inputs

        #Unnormalized Prob
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims = True))

        #Normalized each sample
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims = True)
        self.output = probabilities

    def backward(self, dvalues):
        
        self.dinputs = np.empty_like(dvalues)

        for index, (single_output, single_dvalues) in enumerate(zip(self.output, dvalues)):
            #Flatten output array
            single_output = single_output.reshape(-1, 1)
            
            #Calc Jacobian Matrix of output
            jacobian_matrix = np.diagflat(single_output) - np.dot(single_output, single_output.T)

            #Calc samplewise Grad
            self.dinputs[index] = np.dot(jacobian_matrix, single_dvalues)
    
    #Calculates prediction for outputs
    def predictions(self, outputs):
        return np.argmax(outputs, axis=1)

class Activation_Sigmoid:

    def forward(self, inputs, training):
        #Save input and calculate/save output of sigmoid function
        self.inputs = inputs
        self.output = 1 /(1 + np.exp(-inputs))

    def backward(self, dvalues):
        #Derivative - calculates from output of the sigmoid function
        self.dinputs = dvalues * (1 - self.output) * self.output
    
    #Calculates prediction for outputs
    def predictions(self, outputs):
        return (outputs > 0.5)*1

class Activation_Linear():

    def forward(self, inputs, training):
        #Remeber Values
        self.inputs = inputs
        self.output = inputs

    def backward(self, dvalues):
        #Derivative is 1 => 1 * dvalues = dvalues
        self.dinputs = dvalues.copy()
    
    #Calculates prediction for outputs
    def predictions(self, outputs):
        return outputs

class Loss:
    #Remember Trainable layers
    def remember_trainable_layers(self, trainable_layers):
        self.trainable_layers = trainable_layers

    def regularization_loss(self):

        # 0 by default
        regularization_loss = 0
        
        for layer in self.trainable_layers:
            #L1 regularization- weights calc only when factor greater than 0
            if layer.weight_regularizer_l1 > 0:
                regularization_loss += layer.weight_regularizer_l1 * np.sum(np.abs(layer.weights))
            
            # L2 regularization - weights
            if layer.weight_regularizer_l2 > 0:
                regularization_loss += layer.weight_regularizer_l2 * np.sum(layer.weights * layer.weights)
            
            #L1 regularization- bias calc only when factor greater than 0
            if layer.bias_regularizer_l1 > 0:
                regularization_loss += layer.bias_regularizer_l1 * np.sum(np.abs(layer.biases))
            
            # L2 regularization - bias
            if layer.bias_regularizer_l2 > 0:
                regularization_loss += layer.bias_regularizer_l2 * np.sum(layer.biases * layer.biases)
        

        return regularization_loss

    #Calculate the data and regularization losses give model output and ground truth values
    def calculate(self, output, y, *, include_regularization=False):

        #Calculate sample losses
        sample_losses = self.forward(output, y)

        #Calculate mean loss
        data_loss = np.mean(sample_losses)

        #Add accumulated sum of losses and sample count
        self.accumulated_sum += np.sum(sample_losses)
        self.accumulated_count += len(sample_losses)

        #If just data loss - return it
        if not include_regularization:
            return data_loss

        #Return the data and regularization losses
        return data_loss, self.regularization_loss()
    
    def calculate_accumulated(self, *, include_regularization=False):

        #Calculate mean loss
        data_loss = self.accumulated_sum/self.accumulated_count

        
         #If just data loss - return it
        if not include_regularization:
            return data_loss

        #Return the data and regularization losses
        return data_loss, self.regularization_loss()

    #Resets accumulated loss variables
    def new_pass(self):
        self.accumulated_sum = 0
        self.accumulated_count = 0

class Loss_CategoricalCrossentropy(Loss):
    def forward(self, y_pred, y_true):
        #Number of samples in a batch
        samples = len(y_pred)

        #Clipped data to prevent division by 0. Both sides clipped so no bias 
        y_pred_clipped = np.clip(y_pred, 1e-7, 1-1e-7)

        #Prob for target values
        if len(y_true.shape) == 1: #If categorical labels 
            correct_confidences = y_pred_clipped[range(samples), y_true]
        
        elif len(y_true.shape) == 2: #If one-hot encoded labels
            correct_confidences = np.sum(y_pred_clipped * y_true, axis=1)

        #Losses
        negative_log_likelihoods = - np.log(correct_confidences)
        return negative_log_likelihoods

    def backward(self, dvalues, y_true):
        #Number of samples
        samples = len(dvalues)

        #Number of labels in a sample
        labels = len(dvalues[0])

        #If labels sparse, convert into one-hot vector
        if len(y_true.shape) == 1:
            y_true = np.eye(labels)[y_true]


        #Calc Grad
        self.dinputs = -y_true / dvalues

        #Normalize Grad
        self.dinputs = self.dinputs / samples

class Loss_MeanSquaredError(Loss): #L2 Loss

    def forward(self, y_pred, y_true):

        #Calculate loss
        sample_losses = np.mean((y_true - y_pred)**2, axis=-1)

        #Return losses
        return sample_losses

    def backward(self, dvalues, y_true):

        #Number of samples
        samples = len(dvalues)
        
        #Number of outputs in every sample.
        outputs = len(dvalues[0])

        #Gradient on values
        self.dinputs = -2 * (y_true - dvalues) / outputs

        #Normalize gradient
        self.dinputs = self.dinputs / samples

class Loss_MeanAbsoluteError(Loss): #L1 Loss

    def forward(self, y_pred, y_test):

        #Calculate loss
        sample_losses = np.mean(np.abs(y_true - y_pred), axis = -1)

        return sample_losses

    def backward(self, dvalues, y_pred):

        #Number of samples
        samples = len(dvalues)
        
        #Number of outputs in every sample.
        outputs = len(dvalues[0])

        #Calculate Gradients
        self.dinputs = np.sign(y_true - dvalues) / outputs

        #Normalized Gradients
        self.dinputs = self.dinputs / samples

class Activation_Softmax_Loss_CategoricalCrossentropy():


    def backward(self, dvalues, y_true):

        #Num of samples
        samples = len(dvalues)

        #If one-hot encoded, turn into discrete values
        if len(y_true.shape) == 2:
            y_true = np.argmax(y_true, axis=1)

        #Hard copy
        self.dinputs = dvalues.copy()

        #Calculate Gradient 
        self.dinputs[range(samples), y_true] -= 1

        #Normalize Gradient
        self.dinputs = self.dinputs / samples

class Loss_BinaryCrossentropy(Loss):

    def forward(self, y_pred, y_true):

        #Clip data to prevent division by 0 and clip bot sides to no drage mean towards any value
        y_pred_clipped = np.clip(y_pred, 1e-7, 1-1e-7)

        #Calculate sample-wise loss
        sample_losses = -(y_true * np.log(y_pred_clipped) + (1-y_true) * np.log(1-y_pred_clipped))

        sample_losses = np.mean(sample_losses, axis=-1)

        return sample_losses

    def backward(self, dvalues, y_true):

        #Number of samples
        samples = len(dvalues)

        #Number of outputs in every sample
        outputs = len(dvalues[0])

        #Clip data to prevent division by 0 and clip bot sides to no drage mean towards any value
        clipped_dvalues = np.clip(dvalues, 1e-7, 1-1e-7)

        #Calculate Gradient
        self.dinputs = -(y_true / clipped_dvalues - (1 - y_true) / (1 - clipped_dvalues))

        #Normalize gradient
        self.dinputs = self.dinputs / samples

class Optimizer_SGD():
    
    #Initalize Optimizer. Default learning rate = 1
    def __init__(self, learning_rate=1.0, decay = 0.0, momentum = 0.0):
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iteration = 0
        self.momentum = momentum

    #Call before any parameter updates
    def pre_update_params(self):
        if self.decay:
            self.current_learning_rate = self.learning_rate * (1.0 / (1.0 + self.decay * self.iteration))

    #Update Parameters
    def update_params(self, layer):
        
        #If momentum is being used
        if self.momentum:
            
            #If layer does not have momentume arrays, create them.
            if not hasattr(layer, 'weight_momentums'):
                layer.weight_momentums = np.zeros_like(layer.weights)
                layer.bias_momentums = np.zeros_like(layer.biases)
            
            #Build weight updates with momentum
            #Takes prev updates multipled by the retain factor and update with current gradients
            weights_update =    self.momentum * layer.weight_momentums \
                                - self.current_learning_rate * layer.dweights
            layer.weight_momentums = weights_update
            
            #Builds bias updates
            bias_updates = self.momentum * layer.bias_momentums - self.current_learning_rate * layer.dbiases
            layer.bias_momentums = bias_updates

        #Vanilla SGD
        else:
            weights_update  = -self.current_learning_rate * layer.dweights
            bias_updates    = -self.current_learning_rate * layer.dbiases
        

        #Updates weights and biases with momentum or non momentum
        layer.weights   += weights_update
        layer.biases    += bias_updates

    #Call once after any parameter updates
    def post_update_params(self):
        self.iteration += 1

class Optimizer_Adagrad():
    
    #Initalize Optimizer. Default learning rate = 1
    def __init__(self, learning_rate=1.0, decay=0.0, epsilon=1e-7):
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iteration = 0
        self.epsilon = epsilon

    #Call before any parameter updates
    def pre_update_params(self):
        if self.decay:
            self.current_learning_rate = self.learning_rate * (1.0 / (1.0 + self.decay * self.iteration))

    #Update Parameters
    def update_params(self, layer):
        
      
        #If layer does not have cache arrays, create them.
        if not hasattr(layer, 'weight_cache'):
            layer.weight_cache = np.zeros_like(layer.weights)
            layer.bias_cache = np.zeros_like(layer.biases)
        
        #Update cache with squared current gradients
        layer.weight_cache += layer.dweights**2
        layer.bias_cache += layer.dbiases**2

        #Vanilla SGD parameter update + normalization with squared rooted cache
        layer.weights   +=  -self.current_learning_rate * layer.dweights \
                            / (np.sqrt(layer.weight_cache) + self.epsilon)
        layer.biases    +=  -self.current_learning_rate * layer.dbiases \
                            / (np.sqrt(layer.bias_cache) + self.epsilon) 
        

    #Call once after any parameter updates
    def post_update_params(self):
        self.iteration += 1

class Optimizer_RMSprop():
    
    #Initalize Optimizer. Default learning rate = 1
    def __init__(self, learning_rate=0.001, decay=0.0, epsilon=1e-7, rho=0.9):
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iteration = 0
        self.epsilon = epsilon
        self.rho = rho

    #Call before any parameter updates
    def pre_update_params(self):
        if self.decay:
            self.current_learning_rate = self.learning_rate * (1.0 / (1.0 + self.decay * self.iteration))

    #Update Parameters
    def update_params(self, layer):
        
      
        #If layer does not have cache arrays, create them.
        if not hasattr(layer, 'weight_cache'):
            layer.weight_cache = np.zeros_like(layer.weights)
            layer.bias_cache = np.zeros_like(layer.biases)
        
    

        #Update cache with squared current gradients
        layer.weight_cache   = self.rho * layer.weight_cache + \
                                (1-self.rho) * layer.dweights ** 2
        
        layer.bias_cache    =  self.rho * layer.bias_cache + \
                                (1-self.rho) * layer.dbiases ** 2

        #Vanilla SGD parameter update + normalization with squared rooted cache
        layer.weights   +=  -self.current_learning_rate * layer.dweights \
                            / (np.sqrt(layer.weight_cache) + self.epsilon)
        layer.biases    +=  -self.current_learning_rate * layer.dbiases \
                            / (np.sqrt(layer.bias_cache) + self.epsilon)  

    #Call once after any parameter updates
    def post_update_params(self):
        self.iteration += 1

class Optimizer_Adam():
    
    #Initalize Optimizer. Default learning rate = 1
    def __init__(self, learning_rate=0.001, decay=0.0, epsilon=1e-7, beta_1=0.9, beta_2=0.999):
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iteration = 0
        self.epsilon = epsilon
        self.beta_1 = beta_1
        self.beta_2 = beta_2

    #Call before any parameter updates
    def pre_update_params(self):
        if self.decay:
            self.current_learning_rate = self.learning_rate * (1.0 / (1.0 + self.decay * self.iteration))

    #Update Parameters
    def update_params(self, layer):

        #If layer does not have cache arrays, create them.
        if not hasattr(layer, 'weight_cache'):
            layer.weight_momentums  = np.zeros_like(layer.weights)
            layer.weight_cache      = np.zeros_like(layer.weights)
            layer.bias_momentums    = np.zeros_like(layer.biases)
            layer.bias_cache        = np.zeros_like(layer.biases)
        
        #Update momentum with current gradients
        layer.weight_momentums  =   self.beta_1 * layer.weight_momentums + \
                                    (1 - self.beta_1) * layer.dweights

        layer.bias_momentums    =   self.beta_1 * layer.bias_momentums + \
                                    (1 - self.beta_1) * layer.dbiases

        #Correct Momentum with current gradients
        weight_momentums_corrected =  layer.weight_momentums / \
                                            (1 - self.beta_1 ** (self.iteration + 1))

        bias_momentums_corrected =  layer.bias_momentums / \
                                    (1 - self.beta_1 ** (self.iteration + 1))

        #Update cache with squared current gradients
        layer.weight_cache  =   self.beta_2 * layer.weight_cache + \
                                (1 - self.beta_2) * layer.dweights ** 2

        layer.bias_cache    =   self.beta_2 * layer.bias_cache + \
                                (1 - self.beta_2) * layer.dbiases ** 2
        
        #Get corrected cache
        weight_cache_corrected =    layer.weight_cache / \
                                    (1 - self.beta_2 ** (self.iteration + 1))
        bias_cache_corrected =      layer.bias_cache / \
                                    (1 - self.beta_2 ** (self.iteration + 1)) 

        #Vanilla SGD parameter update + normalization with square rooted cache
        layer.weights   +=  -self.current_learning_rate * weight_momentums_corrected \
                            / (np.sqrt(weight_cache_corrected) + self.epsilon)
        layer.biases    +=  -self.current_learning_rate * bias_momentums_corrected \
                            / (np.sqrt(bias_cache_corrected) + self.epsilon) 

       

    #Call once after any parameter updates
    def post_update_params(self):
        self.iteration += 1

class Accuracy():

    def calculate(self, predictions, y):

        #Get the comparison results
        comparisons = self.compare(predictions,y)

        #Calculate an accuracy 
        accuracy = np.mean(comparisons)

        #Add accumulated sum of losses and sample count
        self.accumulated_sum += np.sum(comparisons)
        self.accumulated_count += len(comparisons)

        #Return accuracy
        return accuracy

    def calculate_accumulated(self):

        #Calculate mean loss
        accuracy = self.accumulated_sum/self.accumulated_count

        #Return the data and regularition losses
        return accuracy

    #Resets accumulated loss variables
    def new_pass(self):
        self.accumulated_sum = 0
        self.accumulated_count = 0

class Accuracy_Regression(Accuracy):

    def __init__(self):
        self.precision = None
    
    #Calculates the precision value based on passed in ground truth
    def init(self, y, reinit=False):
        if self.precision is None or reinit: 
            self.precision = np.std(y) / 250
    
    #Compares predictions to the ground truth value
    def compare(self, predictions, y):
        return np.abs(predictions - y) < self.precision

class Accuracy_Categorical(Accuracy):

    def __init__(self, *, binary=False):
        self.binary = binary

    #No intialization is needed
    def init(self, y):
        pass
    #Compares predictions to the ground truth values
    def compare(self, predictions, y):
        if not self.binary and len(y.shape) ==2:    
            y = np.argmax(y, axis=1)
        return predictions == y

class Model:

    def __init__(self):
        #Creates a list of network objects
        self.layers = []
        #Softmax classifier's output object
        self.softmax_classifier_output = None
    
    #Loads and returns a model
    @staticmethod
    def load(path):

        #Open file in the binary-read mode, load a model
        with open(path, 'rb') as f:
            model = pickle.load(f)

        #Return a model
        return model

    #Adds a object to the model
    def add(self, layer):
        self.layers.append(layer)
    
    def set(self, *, loss=None, optimizer=None, accuracy=None):
        if loss is not None:
            self.loss = loss
        
        if optimizer is not None:
            self.optimizer = optimizer
        
        if accuracy is not None:
            self.accuracy = accuracy
    
    def finalize(self):

        #Create and set the input layer
        self.input_layer = Layer_Input()

        #Count all the objects
        layer_count = len(self.layers)

        #Intialize a list containing trainable layers
        self.trainable_layers = []

        #Iterate over the object
        for i in range(layer_count):

            #If it's the first layer, the previous layer object is the input layer
            if i == 0:
                self.layers[i].prev = self.input_layer
                self.layers[i].next = self.layers[i+1]
            #All layers except for the first and the last            
            elif i < layer_count - 1:
                self.layers[i].prev = self.layers[i-1]
                self.layers[i].next = self.layers[i+1]
            #The last layer - the next object is the loss
            #Also saves aside the reference to the last object whose output is the model's output
            else:
                self.layers[i].prev = self.layers[i-1]
                self.layers[i].next = self.loss
                self.output_layer_activation = self.layers[i]

            #If the layer contains an attribute called "weights", it trainable
            #If so, add it to the list of trainable layers
            if hasattr(self.layers[i], 'weights'):
                self.trainable_layers.append(self.layers[i])

        #Update loss object with trainable layers
        if self.loss is not None:
            self.loss.remember_trainable_layers(self.trainable_layers)

        #IF output activation is Softmax and loss function is Categorical Cross Entropy
        #Create an object of combined activation and loss function containing faster gradient calculation
        if  isinstance(self.layers[-1], Activation_Softmax) and \
            isinstance(self.loss, Loss_CategoricalCrossentropy):
            #Create an object of combined activation and loss functions
            self.softmax_classifier_output = \
            Activation_Softmax_Loss_CategoricalCrossentropy()
            
    #Train the model
    def train(self, X, y, * , epochs=1, batch_size=None, print_every=1, validation_data=None):

        #Intialize the model
        self.accuracy.init(y)
        
        #Default value if batch size is not set
        train_steps =1

        #Calculates Number of steps
        if batch_size is not None:
            train_steps = len(X) // batch_size
            
            #Dividing rounds down. If there are some remaining data, but not a full batch, it won't include it.
            #Add '1' to include this not full batch
            if train_steps * batch_size < len(X):
                train_steps += 1
            
           

        #Main training loop
        for epoch in range(1, epochs+1):
            
            #Print epoch number
            print(f'epoch: {epoch}')
            
            #Reset accumulated values in loss and accuracy objects
            self.loss.new_pass()
            self.accuracy.new_pass()

            for step in range(train_steps):


                if batch_size is None:
                    batch_X = X
                    batch_y = y
                
                else:
                    batch_X = X[step*batch_size:(step+1)*batch_size]
                    batch_y = y[step*batch_size:(step+1)*batch_size]  

                # Perform the forward pass
                output = self.forward(batch_X, training=True)


                # Calculate loss
                data_loss, regularization_loss = \
                    self.loss.calculate(output, batch_y,
                                        include_regularization=True)
                                        
                loss = data_loss + regularization_loss

                # Get predictions and calculate an accuracy
                predictions = self.output_layer_activation.predictions(
                                output)
                accuracy = self.accuracy.calculate(predictions, batch_y)

                # Perform backward pass
                self.backward(output, batch_y)

                # Optimize (update parameters)
                self.optimizer.pre_update_params()
                for layer in self.trainable_layers:
                    self.optimizer.update_params(layer)
                self.optimizer.post_update_params()


                #Print a summary
                if not step % print_every or step == train_steps -1:
                    print(  f'step: {step}, ' +
                            f'acc: {accuracy:.3f}, ' +
                            f'loss: {loss:.3f} (' +
                            f'data_loss: {data_loss:.3f}, ' +
                            f'reg_loss: {regularization_loss:.3f}), ' +
                            f'lr: {self.optimizer.current_learning_rate}')

            epoch_data_loss, epoch_regularization_loss = \
                self.loss.calculate_accumulated(include_regularization=True)
            epoch_loss = epoch_data_loss + epoch_regularization_loss
            epoch_accuracy = self.accuracy.calculate_accumulated()

           
            print(  f'training, ' +
                    f'acc: {epoch_accuracy:.3f}, ' +
                    f'loss: {epoch_loss:.3f} (' +
                    f'data_loss: {epoch_data_loss:.3f}, ' +
                    f'reg_loss: {epoch_regularization_loss:.3f}), ' +
                    f'lr: {self.optimizer.current_learning_rate}')
            
            #If there is the validation data
            if validation_data is not None:
                #Evaluate the mode
                self.evaluate(*validation_data, batch_size=batch_size)
               
    def forward(self, X, training):

        #Call forward method on the input layer
        #This will set the output property that the first layer in "prev" object is expecting
        self.input_layer.forward(X, training)

        #Call forward method on every object in a chain and pass the output of the previous layer as a parameter
        for layer in self.layers:
            layer.forward(layer.prev.output, training)

        #Return the last layer from the list
        return layer.output
    
    def backward(self, output, y):
        
        if self.softmax_classifier_output is not None:

            #First call backward method on the combined activation/loss this will set dinputs property
            self.softmax_classifier_output.backward(output, y)

            #Since we'll not call backward method of the last layer which is Softmax activation
            #as we used combined activation/loss objects, let's set dinputs in object
            self.layers[-1].dinputs = self.softmax_classifier_output.dinputs

            #Call backward method going through all the objects but last in reversed order passing dinputs as a parameter
            for layer in reversed(self.layers[:-1]):
                layer.backward(layer.next.dinputs)
            return
        
        
        
        #First call backward method on loss. 
        #This will set dinputs property that the last layer will try to access shortly
        self.loss.backward(output, y)

        #Call in reverse the backward method going through all the objects passing dinputs as a prameter
        for layer in reversed(self.layers):
            layer.backward(layer.next.dinputs)

    def evaluate(self, X_val, y_val, *, batch_size=None):
        
        #Default value if batch size is not being set
        validation_steps = 1

        #Calculates the number of steps
        if batch_size is not None:
            validation_steps = len(X_val) // batch_size
            
            #Dividing rounds down. If there are some remaining data, but not a full batch, it won't include it.
            #Add '1' to include this not full batch
            if validation_steps * batch_size < len(X_val):
                validation_steps += 1
        
        #Reset accumulated values in loss and accuracy objects
        self.loss.new_pass()
        self.accuracy.new_pass()

        for step in range(validation_steps):
            
            #If batch size is not set, train using 1 step and full dataset
            if batch_size is None:
                batch_X = X_val
                batch_y = y_val
            #Otherwise slice a batch
            else:
                batch_X = X_val[step*batch_size:(step+1)*batch_size]
                batch_y = y_val[step*batch_size:(step+1)*batch_size] 


            #Perform the forward pass
            output = self.forward(X_val, training=False)

            #Calculate the loss
            self.loss.calculate(output,y_val)

            #Get predictions and calculate an accuracy
            predictions = self.output_layer_activation.predictions(output)

            accuracy = self.accuracy.calculate(predictions, y_val)

        #Get and print validation loss and accuracy
        validation_loss = self.loss.calculate_accumulated()
        validation_accuracy = self.accuracy.calculate_accumulated()
        #Print a summary
        print(  f'validation, '+
                f'acc: {validation_accuracy:.3f}, '+
                f'loss:{validation_loss:.3f}')

    def get_parameters(self):
        #Create a list of parameters
        paramaters = []

        #Iterable trainable layers and get their parameters
        for layer in self.trainable_layers:
            paramaters.append(layer.get_parameters())

        #Return a list
        return paramaters

    #Updates the model with new parameters
    def set_parameters(self, paramaters):
        #Iterate over the parameter and layer and 
        #update each layers with each set of the parameters
        for paramater_set, layer in zip(paramaters, self.trainable_layers):
            layer.set_parameters(*paramater_set)

    #Saves the parameters to a file
    def save_parameters(self, path):
        #Open a file in the binary-write mode and save the parameters to it
        with open(path, 'wb') as f:
            pickle.dump(self.get_parameters(), f)
    
    #Loads the wights and updates a model instance with them
    def load_parameters(self, path):

        #open file in binary-read mode, load weights and update trainble layers
        with open(path, 'rb') as f:
            self.set_parameters(pickle.load(f))

    #Save the model
    def save(self, path):

        #Make a deep copy of current model instance
        model = copy.deepcopy(self)

        #Reset accumulated values in loss and accuracy objects
        model.loss.new_pass()
        model.accuracy.new_pass()

        #Remove data from input layer and graidents from the loss object
        model.input_layer.__dict__.pop('output', None)
        model.loss.__dict__.pop('dinputs', None)

        #For each layer remove inputs, output, and dinputs properties
        for layer in model.layers:
            for property in ['inputs', 'output', 'dinputs', 'dweights', 'dbiases']:
                layer.__dict__.pop(property, None)
        
        #open a file in the binary-write mode and save the model
        with open(path, 'wb') as f:
            pickle.dump(model, f)

    def predict(self, X, *, batch_size=None):

        #Default value if batch size is not being set
        prediction_steps = 1

        #Calculates number of steps
        if batch_size is not None:
            prediction_steps = len(X // batch_size)

            #Dividing rounds down. If there are some remaining data, but not a full batch, it won't include it.
            #Add '1' to include this not full batch
            if prediction_steps * batch_size < len(x):
                prediction_steps += 1
        
        #Model outputs
        output = []

        for step in range(prediction_steps):


            #If batch size is not set, train using 1 step and full dataset
            if batch_size is None:
                batch_X = X
            #Otherwise slice a batch
            else:
                batch_X = X[step*batch_size:(step+1)*batch_size]

            
            #Perform the forward pass
            batch_output =  self.forward(batch_X, training=False)

            #Append batch prediction to the list of predictions
            output.append(batch_output)

        return np.vstack(output)
     
#Load a MNIST dataset
def load_mnist_dataset(dataset, path):

    #Scan all the directories and create a list of labels
    labels = os.listdir(os.path.join(path, dataset))

    #Create lists for samples and labels
    X = []
    y = []

    #For each label in the folder
    for label in labels:
        #And for each image in given folder
        for file in os.listdir(os.path.join(path, dataset, label)):
            #Read the image
            image = cv2.imread(os.path.join(path, dataset, label, file), cv2.IMREAD_UNCHANGED)
            #And append it and a label to the lists
            X.append(image)
            y.append(label)

    #Convert the data to numpy arrays and return
    return np.array(X), np.array(y).astype('uint8')

def create_data_mnist(path):

    #Load both sets seperately
    X, y = load_mnist_dataset('train', path)
    X_test, y_test = load_mnist_dataset('test', path)
    
    #And return all the data
    return X, y, X_test, y_test




