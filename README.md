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

    ### 🚀 Activation Functions 
    
    | Function | Mathematical Formula | Output Range | Ideal Use Case |
    | :--- | :---: | :---: | :--- |
    | **Linear** | $f(x) = x$ | $(-\infty, \infty)$ | Regression Output |
    | **Sigmoid** | $f(x) = \frac{1}{1 + e^{-x}}$ | $(0, 1)$ | Binary Classification |
    | **Tanh** | $f(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}}$ | $(-1, 1)$ | Hidden Layers (Zero-centered) |
    | **ReLU** | $f(x) = \max(0, x)$ | $[0, \infty)$ | Standard Hidden Layers |
    | **Leaky ReLU** | $f(x) = \max(\alpha x, x)$ | $(-\infty, \infty)$ | Fixing "Dead Neurons" |
    | **ELU** | $f(x) = \begin{cases} x & x > 0 \\ \alpha(e^x - 1) & x \le 0 \end{cases}$ | $(-\alpha, \infty)$ | Improved mean activations |
    | **GELU** | $f(x) = x\Phi(x)$ | $\approx [-0.17, \infty)$ | Transformers & LLMs |
    | **Softmax** | $\sigma(\mathbf{z})_i = \frac{e^{z_i}}{\sum_{j=1}^K e^{z_j}}$ | $(0, 1)$ | Multi-class Classification |
    

IMAGEs on neural network(Multiple Neurons):

    Say an image is fed as input.It is known to have a shape of  128X128 meaning the 784 pixels .So they are flattened and 784 nodes form the input

as **SKLEARN** IS FOR ML,**KERAS** HELPS FOR DL

    Keras also has datasets within. keras.datasets.fashion_mnist.load_data() returns tuples (train_images,train_labels) and (test_images,test_labels)
    train_images will have features.train_labels have labels.

Decide on the layers.Compile them.
An optimizer decides how the weights should be reconfigured.
Model building is done.

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

ImageNet is a dataset 

  .
