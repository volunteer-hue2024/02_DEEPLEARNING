# 02 _ DEEP LEARNING 
     Unlike in ML here it is a single architecture able to do many differnt things.
     
Neural Network :

    Inspired from human brain, the neural n/w containing many neurons, each able to make or break connections as in brain.<br>
        
    Neuroplasticity: It involves synaptic reorganization, where connections between neurons become stronger or weaker based on usage.
   
Perceptron :
        
        One single neuron is called perceptron.A single neuron has multiple inputs to it,an activation function causing the primary output value.
        
To a neuron what is the input data? What is the output? 
        One rowset of **Features** form the input data.Output is the **Label** 

       Say dataset has 3 features ,age ,credit score,salary ; Fraud 0,1 is the prediction value.So this is a classification prob.
       1st set of input is from the first dataset row.Based on how close its prediction was (to actual result) the neuron learns from it.
        If it was a single input node then only one weight need to be trained bcoz y =mx+c...else if more than one feature is present y=m(x1+ x2+         x3 ) + C , 3 weights are to be tuned. Each neuron has a bias too to be trained.     
        Next, 2nd row is fed.
        The neuron has one bias always , and one weight each for every feature.
        Activation layer decides what data has to be sent to the next layer.
        when the input is just one feature it is in effect an equation of line..where weight is the slope and bias is the intercept.The nucleous          or the brain can have many kinds of functions,each giving differnt output threshold

Say a neural network has  3 input nodes and 2 hidden layers of 6 neurons each . Output layer has one neuron.How many weights are to be trained?

    at the first hidden layer 3weights + 1 bias  for a neuron. Similarly 6 neurons at this layer ie = 6*(3 weights + 1 bias)  = 24 parameters

    at the second hidden layer 6 weights +1 bias for a neuron .Similary 6 neurons totally at this layer viz 6*(6 weights + 1) bias  = 42 parameters.

    at the output layer 6 weights +1 bias 
    Totally 24+42+7=73 parameters to be trained.
    
       
What are the possible activation functions at a neuron?

               ### 🧠 Activation Functions Reference

| Function | Formula | Output Range | Common Use Case |
| :--- | :--- | :--- | :--- |
| **Linear** | $f(x) = x$ | $(-\infty, \infty)$ | Regression output layers |
| **Sigmoid** | $f(x) = \frac{1}{1 + e^{-x}}$ | $(0, 1)$ | Binary Classification |
| **Tanh** | $f(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}}$ | $(-1, 1)$ | Hidden layers (Zero-centered) |
| **ReLU** | $f(x) = \max(0, x)$ | $[0, \infty)$ | Standard for hidden layers |
| **Leaky ReLU** | $f(x) = \max(\alpha x, x)$ | $(-\infty, \infty)$ | Prevents "Dead Neuron" problem |
| **GELU** | $f(x) = x \Phi(x)$ | $\approx [-0.17, \infty)$ | Transformers (GPT, BERT) |
| **Softmax** | $\sigma(\mathbf{z})_i = \frac{e^{z_i}}{\sum e^{z_j}}$ | $(0, 1)$ | Multi-class Classification |

HOW ARE IMAGEs PROCESSED on neural network(Multiple Neurons):

    Say an image is fed as input.It is known to have a shape of  128X128 meaning it has 784 pixels .So they are flattened and 784 nodes form the input.

as **SKLEARN** IS FOR ML,**KERAS** HELPS FOR DL

    Keras also has datasets within. keras.datasets.fashion_mnist.load_data() returns tuples (train_images,train_labels) and (test_images,test_labels)
    train_images will have features.train_labels have labels.

Decide on the layers.Compile them.
An optimizer decides how the weights should be reconfigured.
Model building is done.

Code 

               import numpy as np
               import pandas as pd
               import tensorflow as tf
               from tensorflow import keras
               import matplotlib.pyplot as plt
               import plotly.graph_objects as go
               from plotly.subplots import make_subplots
               
               # Load Dataset
               (train_images, train_labels), (test_images, test_labels) = keras.datasets.fashion_mnist.load_data()
               
               # Normalize Data
               train_images = train_images / 255.0
               test_images = test_images / 255.0
               
               # Reshape Data
               train_images = train_images.reshape((train_images.shape[0], 28, 28, 1))
               test_images = test_images.reshape((test_images.shape[0], 28, 28, 1))
               
               # One-Hot Encode Labels
               train_labels_one_hot = keras.utils.to_categorical(train_labels, num_classes=10)
               test_labels_one_hot = keras.utils.to_categorical(test_labels, num_classes=10)
               
               # Verify Dataset Shapes
               print("Training images shape:", train_images.shape)
               print("Testing images shape:", test_images.shape)
               print("Training labels shape:", train_labels_one_hot.shape)
               print("Testing labels shape:", test_labels_one_hot.shape)
               
               # Basic ANN Model
               ann_model = keras.Sequential([
                   keras.layers.Flatten(input_shape=(28, 28, 1)),
                   keras.layers.Dense(128, activation='relu'),
                   keras.layers.Dense(64, activation='relu'),
                   keras.layers.Dense(10, activation='softmax')
               ])
               
               ann_model.compile(optimizer='adam',
                                 loss='categorical_crossentropy',
                                 metrics=['accuracy'])
               
               ann_model.summary()
               
               
### 📉 Loss Functions Reference

| Category | Loss Function | Mathematical Formula | Key Use Case |
| :--- | :--- | :--- | :--- |
| **Regression** | **MSE** (Mean Squared Error) | $L = \frac{1}{n} \sum (y_i - \hat{y}_i)^2$ | Predicting continuous values (House prices, Temp). |
| **Regression** | **MAE** (Mean Absolute Error) | $L = \frac{1}{n} \sum \vert y_i - \hat{y}_i \vert$ | Regression robust to outliers. |
| **Binary Class** | **Binary Cross-Entropy** | $L = -[y \log(\hat{y}) + (1-y) \log(1-\hat{y})]$ | Two-choice classification (Spam vs. Not Spam). |
| **Multi-Class** | **Categorical Cross-Entropy** | $L = -\sum y_i \log(\hat{y}_i)$ | Multi-class (Dog vs. Cat vs. Bird) with One-Hot labels. |
| **Multi-Class** | **Sparse Categorical CE** | $L = -\sum y_i \log(\hat{y}_i)$ | Same as above, but uses integer labels (0, 1, 2...). |
| **Ranking/Embedding**| **Hinge Loss** | $L = \max(0, 1 - y \cdot \hat{y})$ | Support Vector Machines (SVM) and some GANs. |
| **Advanced** | **Huber Loss** | Combination of MSE and MAE | Robust regression that handles outliers gracefully. |

What is Entropy?

Entropy is a measure of the uncertainty or randomness in a dataset.

    The Concept: If a result is certain (e.g., a biased coin that always lands heads), the entropy is 0. 
    If a result is completely unpredictable (e.g., a fair 6-sided die), the entropy is high.

Cross-Entropy
        
        Cross-Entropy measures the difference between two probability distributions:the True distribution (P) and the Predicted distribution (Q).

What is an epoch?

     One iteration of the whole data..each epoch going on means the training is going on.
     Each epoch shows the accuracy,loss,validation accuracy and validation loss.
     
     If the same model is executed over again ,the model will start from where it was left.(since the model is not rebuilt.)
     Weights are turning better each time.
     
What is Batch Size?
          
          : if batch_size is 1 , for 60000 images the weights will be updated 60000 times.
          If batch_size is 6000 ,the weights will be updated 10 times
          **Batch size in deep learning is
          a hyperparameter defining the number of training samples processed in one forward/backward pass before updating model weights**
          
ImageNet is a dataset 

  .
