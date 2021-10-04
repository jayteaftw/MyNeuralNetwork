# My Neural Network

A Neural Network built from scratch.
The model was trained on the MNIST Fashion Dataset.
Project is based on the Book: "Neural Network from Scratch in Python"

## File Descriptions
**MNNLibrary.py**: Contains base functions such as Dense Layers, Activation functions, Loss functions, Accuracy functions, Optimizer functions and an all encompassed Model function.

**train.py**: Trains the model on a preset small architecture([1,256]->[256,256]->[256,10] *Subject to Change*)

**predict.py**: Predicts an input image on a give model

**evaluate.py**: Evaluates a given model on the test set data for MNIST Fashion dataset

**show.py**: Displays the input image in a 28 by 28 pixel grayscaled ratio.

## Examples and Findings
I was able to obtain a model with a 89% accuracy on the test model.

Here something interesting outputs I would like to share


**Shirt 1**:
Here is a picture I picked downloaded on the internet of a shirt.

![shirt2](https://user-images.githubusercontent.com/20690770/135779845-d2cf47f5-4639-43d9-8ff9-1ea909700cea.jpeg)

**Terminal**:
 >$ python3 predict.py shirt1.jpg fashion_mnist_89 

 >\>\> T-shrit/top

Shirt 1 is a success! However.....


**Shirt 2**: Picture I took.

!<img src="https://user-images.githubusercontent.com/20690770/135780494-f66f3f23-3967-4e5f-8398-44fe3a45af51.jpg" width=40%>

**Terminal**:

 >$ python3 predict.py shirt2.jpg fashion_mnist_89 

 >\>\> Bag

That does not seem right... However if we use the method show.py, we can get some insight about what is going on.

**Terminal**:

 >$ python3 show.py shirt2.jpg

![Shirt2_Show](https://user-images.githubusercontent.com/20690770/135780828-c8850ea0-d9bf-49af-8681-a7ecded7e83c.png)

Can you tell if that is a shirt? I can't. 
This shows the importance of good data.

**Shirt 3**: Second picture I took.

!<img src="https://user-images.githubusercontent.com/20690770/135781010-c37886e7-20ab-4342-bbd0-d084990b873d.jpg" width=40%>

**Terminal**:

 >$ python3 predict.py shirt3.jpg fashion_mnist_89 

 >\>\> T-shrit/top

Bingo! And lets see the it in a 28x28 grayscale

![Shirt3_Show](https://user-images.githubusercontent.com/20690770/135781274-e8022e95-4aff-4d4c-8a7e-0960d24af6ba.png)

Looks kind of like a shirt. However lets change the shirt slightly

**Shirt 3**: The previous shirt changed slightly near the arm holes

!<img src="https://user-images.githubusercontent.com/20690770/135781346-a49d13ba-9d24-4fda-98e1-a9b32b274beb.jpg" width=40%>


**Terminal**:

 >$ python3 predict.py shirt3_2.jpg fashion_mnist_89
  
 >\>\> Bag

Another bag......... Lets check the show.py file.

![Shirt3_2_Show](https://user-images.githubusercontent.com/20690770/135781482-0ca19809-93dd-40ba-bddc-e607205b103e.png)

I can see the resemblance to a bag. Since this is a Neural Network not a Convolutional neural network, the model is examining each pixel rather than looking for features on the images themselves such as lines and curves which leads to this issue.

Finally, 

**Coat 1**: A picture of a coat I took.

!<img src="https://user-images.githubusercontent.com/20690770/135781865-3380a56e-cb92-47f2-824f-0736e318f4f6.jpg" width=40%>

**Terminal**: 

 >$ python3 predict.py coat1.jpg fashion_mnist_89 
 >
 >\>\> Coat

Looks Good! Lets check the show.py file

![coat1_Show](https://user-images.githubusercontent.com/20690770/135782042-702e9be7-bcb6-4b87-aec5-951e74c1ec85.png)


