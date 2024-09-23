

   
# 201_dl_coutch-01_13092024.py
"""
Part 01:
get_MyLinks:
	print('https://playground.tensorflow.org")
Deep learning
	- Type of machine learning inspired by human brains
	- Structure : Artificial neural networks
	- It needs a massive number of data to be trained
	- It also needs high GPU
	- It takes time to be trained
	- It is the most used
	![](https://www.researchgate.net/profile/Umair-Shahzad/publication/357631533/figure/fig1/AS:1109568985808903@1641553269731/Machine-learning-as-a-subfield-of-artificial-intelligence_Q320.jpg)

Famous frameworks :
		- Tensorflow
		- Keras
		- Pytorch
		- Dl4j
		- Caffe
		- Microsoft cognitive toolkit
	![](https://miro.medium.com/v2/resize:fit:467/1*rJFONqrZEN7y7cwrVt3lXw.jpeg)

Deep learning vs Machine Learning
	- Deep learning will give you a lot of benefits :
		- No dimensionality reduction
		- No feature extraction
		- Deal with structured and un structured data (audio , images , videos , even text)
		- more data more accurate and performance
		- Complex problems
	![](https://images.prismic.io/turing/652ebc26fbd9a45bcec81819_Deep_Learning_vs_Machine_Learning_3033723be2.webp?auto=format%2Ccompress&fit=max&w=3840)

	- But with cost :
		- A lot of Training time
		- High Computational power GPU
		- Huge amount of data
		- More data will lead to saturation

Why not deep learning :
	- Simple problems
	- Acceptable error
	- Small data
	
DL Applications
	- Customer support
	- Medical care
	- Self driving cars
	- Face recognition
	- Object detection
	- Recommendations
	- Robotics

DL steps
	- The network will be layers on neurons
	- First layer : input layer
	- Last layer : output layer
	- In the middle : the hidden layers (for computations , feature extraction)
	- The input data will be a flatten matrix to the first layer (each pixel to a neuron)
	- Neurons for a layer are connected to the neurons of the next layer using channels
	- Each channel assigned a value = weight
	- The neuron value will be multiplied by the weight + the bias (neuron value)
	- The result value will be passed to a threshold value called = activation function
	- The result of the activation function will determine if the particular neurons will get activated or not
	- Activated neuron will pass the data to the neurons of the next layer using the channels
	- And so on the data will be propagated through the network (forward progradation)
	- In the output layer the neuron with highest value (probability) fires the output (predicted)
	- During this process the network will compare the predicted output with the real output to realize the error (backward propagation)
	- Based on the this the weights are being adjusted
	- This process will continue until the the network can predict the output correctly (most of the cases)
	![](https://miro.medium.com/v2/resize:fit:1400/1*OGFvJgMe21_5fCzUUyLwrw.png)

ANN
- Build on top of biological neuron network
- The simplest ann consist of 1 hidden layer
- Many layers = deep neural network
- Each neuron connects to another and has an associated weight and threshold
- The first weights are generated randomly , then be optimized
- If the output of any individual neuron is above the threshold value , that node is activated (send data to the next layer) if not no data will be passed (this feature is not important)
- There are many hidden layer types , most general is dense layer
- All deep learning networks build on top of gradient descent
![](https://miro.medium.com/v2/resize:fit:933/1*wBhuiErzkNMiKa1yklXbZA.png)

Activation function 
- defines how the weighted sum on input is transformed into an output
![](https://media.licdn.com/dms/image/D4D12AQH2F3GJ9wen_Q/article-cover_image-shrink_720_1280/0/1688885174323?e=2147483647&v=beta&t=gFWxErTLLWBc6iRWDxCBRxkdJ7ob24cmjWZAOuKN9o4)

Loss function :
- Find the difference between expected sand the predicted
![](https://miro.medium.com/v2/resize:fit:616/1*N1PyOYeog-vyytRbwEwQCQ.png)

Optimizer :
- Used to change the attributes of the neural network
(weight , batch size , learning rater ) to reduce the loss
- Determine how the network will be updated next epoch
- Weight updating process called backpropagation
![](https://miro.medium.com/v2/resize:fit:1200/1*Vi667n-YgG04HLVoi95ChQ.png)
![](file:///G:/My_Drive/MyPyWork/bilder/optimizer.png)  

Advantages :
- ANN has the ability to learn and model non-linear and complex relationships as many relationships between input and output are non-linear.
- After training, ANN can infer unseen relationships from unseen data, and hence it is generalized.
- Unlike many machine learning models, ANN does not have restrictions on datasets like data should be Gaussian distributed or nay other distribution.

Applications:
- Image Preprocessing and Character Recognition.
- Forecasting.
- Credit rating.
- Fraud Detection.

Layers :
Input Layer
	- Purpose: The input layer is where the neural network receives the data that will be processed.
	- Structure: Each neuron in this layer corresponds to a feature or attribute in the input data.
	- For example, if your dataset has 5 features (age, sex, cholesterol, etc.), the input layer will have 5 neurons.
	- How It Works: The values of the input data (e.g., pixel values in an image, or features in a tabular dataset) are fed into the network via the input layer. No learning
	happens here; it simply passes the data to the next layer.

Hidden Layers
	- Purpose: The hidden layers are where the actual learning happens in the neural network. They transform the input data into useful representations.
	- Structure: A deep learning model can have multiple hidden layers, and each hidden layer can have a different number of neurons. The number of neurons in each hidden layer is a hyperparameter that is usually determined through experimentation.
	- How It Works:
		- Each neuron in the hidden layers performs a weighted sum of the inputs it receives from the previous layer (input layer or another hidden layer).
		- The weighted sum is passed through an activation function (like ReLU, Sigmoid, etc.), introducing non-linearity, enabling the model to learn complex patterns.
		- The hidden layers progressively extract features from the data. Early layers may extract simple features, while deeper layers combine these into more complex patterns.
		- The process is repeated for each hidden layer, with the output from one layer feeding as input to the next.

Output Layer
	- Purpose: The output layer is where the final decision or prediction is made.
	- Structure: The number of neurons in the output layer depends on the task:
	- Classification tasks: For binary classification (e.g., "yes" or "no"), there will be 1 neuron in the output layer. For multi-class classification (e.g., categorizing something into 5 classes), there will be as many neurons as there are classes.
	- Regression tasks: There is typically one neuron in the output layer, representing the predicted continuous value.      
	- How It Works:
		- In a classification task, the neurons in the output layer use an activation function like softmax (for multi-class) or sigmoid (for binary classification) to produce probabilities or output scores.
		- In regression tasks, the activation function may be a linear function (or no activation at all), allowing the output to be a continuous value.
		 ![](https://miro.medium.com/v2/resize:fit:1199/1*N8UXaiUKWurFLdmEhEHiWg.jpeg)

Chalenging Questions:
- How many hidden layers should be selected ?
- How many neurons in each layer should be ?
- Which activation function should be selected ?
- In case of training, what is the number of epochs and batches ?

"""
