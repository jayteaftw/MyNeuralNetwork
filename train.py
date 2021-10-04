from MNNLibrary import *
import sys

if len(sys.argv) < 2:
    print("ERROR! Missing Argument. Command: \"python3 train.py OutputModelName\"")
    sys.exit()
if len(sys.argv) >2:
    print("ERROR! Too Many Arguments. Command: \"python3 train.py OutputModelName\"")
    sys.exit()

#Create dataset,
X, y, X_test, y_test = create_data_mnist('fashion_mnist_images')

#Shuffle dataset training
keys = np.array(range(X.shape[0]))
np.random.shuffle(keys)
X = X[keys]
y = y[keys]

#Scale and reshape samples
X = (X.reshape(X.shape[0], -1).astype(np.float32) - 127.5) / 127.5
X_test  = (X_test.reshape(X_test.shape[0],-1).astype(np.float32) - 127.5) / 127.5


#Instantiate the model
model =Model()

#Add layers
model.add(Layer_Dense(X.shape[1], 256))
model.add(Activation_ReLU())
model.add(Layer_Dense(256, 256))
model.add(Activation_ReLU())
model.add(Layer_Dense(256, 10))
model.add(Activation_Softmax())

#Set loss and optimizer objects
model.set(  loss=Loss_CategoricalCrossentropy(), 
            optimizer=Optimizer_Adam(decay=1e-3),
            accuracy=Accuracy_Categorical()
)

#Finalize the model
model.finalize()

#Train the model
model.train(X, y, validation_data=(X_test, y_test), epochs=20, batch_size=128, print_every=100)

#Saves the model
model.save(sys.argv[1]+'.model')