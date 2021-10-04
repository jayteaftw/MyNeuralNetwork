from MNNLibrary import *
import sys

if len(sys.argv) < 2:
    print("ERROR! Missing Argument(s). Command: \"python3 evaluate.py ModelName\"")
    sys.exit()
if len(sys.argv) > 2:
    print("ERROR! Too Many Arguments. Command: \"python3  evaluate.py ModelName\"")
    sys.exit()

#Create dataset,
X, y, X_test, y_test = create_data_mnist('fashion_mnist_images')

#Scale and reshape samples
X_test  = (X_test.reshape(X_test.shape[0],-1).astype(np.float32) - 127.5) / 127.5

#Load the model
model = Model.load(sys.argv[1])

#Evaluate the model
model.evaluate(X_test, y_test)
