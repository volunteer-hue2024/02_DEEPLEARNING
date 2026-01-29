# station_02_DL
Deep Learning : unlike in ML here a single architecture inspired from human brain (neural n/w) is used for many purposes
Single neurons : (Perceptrons) Key Aspects 

    Neuroplasticity: It involves synaptic reorganization, where connections between neurons become stronger or weaker based on usage.
    A single neuron has multiple inputs to it
    Perceptron : One single neuron is called perceptron

    Say the dataset has 10 features ,the input layer will have 10 input nodes.Output will still be 2,viz 0 or 1.
    
    This is repeated for each row and the model makes a learning.If the number of features is 3 ,the input neuron will be 3.
    The nucleus has one bias always , and one weight each for every feature.
    Activation layer decides what data has to be sent to the next layer.
    when the input is just one feature it is in effect an equation of line..where weight is the slope and bias is the intercept.The nucleous or the brain can have many kinds of functions,each giving differnt output threshold

If Multiple Neurons:- This is a neural network

Say an image is fed as input.It is known to have a shape of 60,000 ,128,128 meaning the input nodes will be 128*128 in count.
An optimizer decides how the weights should be reconfigured.
Model building is over.

NEXT WE TRAIN THE MODEL.

What is an epoch?
One iteration of the whole data..each epoch going on means the training is going on.
Each epoch shows the accuracy,loss,validation accuracy and validation loss.

If the same model is executed over again ,the model will start from where it was left.(since the model is not rebuilt.)
Weights are turning better each time.
Batch Size : if batch_size is 1 , for 60000 images the weights will be updated 60000 times.
If batch_size is 6000 ,the weights will be updated 10 times
**Batch size in deep learning is
a hyperparameter defining the number of training samples processed in one forward/backward pass before updating model weights**

  .
