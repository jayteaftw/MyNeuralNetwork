from MNNLibrary import *
import sys

if len(sys.argv) < 3:
    print("ERROR! Missing Argument(s). Command: \"python3 predict.py FileName ModelName\"")
    sys.exit()
if len(sys.argv) > 3:
    print("ERROR! Too Many Arguments. Command: \"python3 predict.py FileName ModelName\"")
    sys.exit()


#Dictionary for fashion MNIST dataset
fashion_mnist_labels = {
    0:  'T-shrit/top',
    1:  'Trouser',
    2:  'Pullover',
    3:  'Dress',
    4:  'Coat',
    5:  'Sandal',
    6:  'Shirt',
    7:  'Sneaker',
    8:  'Bag',
    9:  'Ankle boot'
}

#Read an image
image_data = cv2.imread(sys.argv[1], cv2.IMREAD_GRAYSCALE)

#Resize to the same size as Fashion MNIST images
image_data = cv2.resize(image_data, (28, 28))

#Invert image colors
image_data = 255 - image_data

#Reshape and scale pixel data
image_data = (image_data.reshape(1, -1).astype(np.float32) - 127.5) / 127.5

#Load the model
model = Model.load(sys.argv[2])

#Predict on the image
confidences = model.predict(image_data)

#Get prediction instead of confidence levels
predictions = model.output_layer_activation.predictions(confidences)

#Get label name from label index
prediction = fashion_mnist_labels[predictions[0]]

print(prediction)
