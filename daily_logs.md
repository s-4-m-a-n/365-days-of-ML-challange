# Daily logs

# Day 1:
🎯 day 1 of #365daysofML

🔥 Tensorflow from basic to advance:
---
TensorFlow is an open-source library developed by Google for numerical computation and machine learning tasks. It provides a comprehensive ecosystem of tools, libraries, and resources that facilitate the development and deployment of machine learning models.


🌐 Resources
---
I am using official documentation of the tensorflow. [visit documentation](https://www.tensorflow.org/guide/basics)
In addition with the, I will be using ChatGPT, medium articles, and stackoverflow to get in-depth insight more precisely.

✅ Daily logs
---
- Walk through the main components of tensorflow 2 which includes Constant, variables, graph.
- learning various tf.math operations

🔑 Keypoints
- 
- A tensor is a generalization of vector and metrics to the potentially higher dimensions. Is is like an np.array.
- Tensor is primary data type in tensorflow which is immutable. tf.constant method is used to create a tensor.
- RaggedTensor, SparseTensor are the other types of tensor, RaggedTensor allows us to store variable length of elements in some axis, where SparseTensor allows us to store sparse matrix memory efficiently.
- A Variable represents a tensor whose values can be changed by running operations in it.
- Tensorflow uses Variable to store model parameters.
- A variable looks and acts like a tensor, in fact variable is a datastructure that is backed by tf.Tensor. 
- We can assign new values in a variable which is not possible in tensor because of its immutable property. 
- Assigining new values in a variable doesnot create a new tensor, instead the existing tensor memory is reused.
- For better performance, tensorflow will by default tries to place tensor and variable to the fastest device compatible with its datatype i.e GPU if available.
- It is also possible to place tensors or variable on one device and do the computation on another device.
- There are two types of execution mode namely, graph execution and eager execution.
- In eager execution, TensorFlow operations are executed immediately as they are called, similar to regular Python code.
- In graph execution, computations are defined as a static computational graph before the actual execution.
- Graph is a data structure that contains a set of tf.Operation and tf.Tensor objects.
- Since, graph is a data structure they can be saved, run and stored without the original python code.
- Graphs are extremely useful and let your TensorFlow run fast, run in parallel, and run efficiently on multiple devices.
- If you are familiar with parse tree, then you view graph as a parse tree.
- Graph tends to offer better performance.
- Grappler is the default graph optimization system in the TensorFlow runtime.  
- tf.function can be used to create a graph in tensorflow.
- tf.function uses a library called AutoGraph (tf.autograph) to convert Python code into graph-generating code.
 
🔗  Github Repository: https://github.com/s-4-m-a-n/365-days-of-ML-challenge/tree/main/tensorflow%20from%20basic%20to%20advance


#365daysofMLchallenge
#machinelearning
#deeplearning
#MLOps
#tensorflow2
#keras


# Day 2:
🎯 day 2 of #365daysofML

✅ Daily logs
---
- learned about gradient and automatic differentiation.
- learn how to create models, layers using tf.Module class
- learning various tf.math operations


🔑 Keypoints
---
 - The gradient provides information about the rate and direction of change of a function with respect to the corresponding variable.
 - There are different techniques to compute gradient which includes sympolic differentiation, neumerical differentiation, automatic differentiation (forward and backward AD)
 - Tensorflow uses reverse auto differentiation to automatically compute the gradient of a function.
 - tf.GradientTape is a powerful API that allows you to compute gradients of operations with respect to variables.
 - what will happen when we assign a new value to the variable and execute the tape.gradient()?
    - The persistence feature allows you to compute multiple gradients within the same tape context, 
    but it doesn't retain information about changes to variables made after the initial gradient computation.
    - TensorFlow's gradient tape records operations in a forward pass and uses those recorded operations to compute gradients during the backward pass. Once you have executed tape.gradient(), the tape is "consumed," and it no longer retains information about subsequent changes to variables.

- A model is, abstractly:
   - A function that computes something on tensors (a forward pass)
   - Some variables that can be updated in response to training
- Most models are made of layers.
- Layers are functions with a known mathematical structure that can be reused and have trainable variables. 
- Modules in TensorFlow provide a means to encapsulate related functionality, such as layers, blocks, or custom operations, into a reusable unit.
- Modules help with code organization, modularity, and reusability, allowing you to encapsulate related functionality into separate units.
- TensorFlow can run saved models without the original Python objects.
- TensorFlow needs to know how to do the computations described in Python, but without the original code. To do this, you can make a graph.

# Day 3
🎯 day 2 of #365daysofML

✅ Daily logs
---
- learned about distributed training in tensorflow.
- learned about training loops
- learning various tf.math operations


🔑 Keypoints

- There are two approaches for distributed training that includes data parallelism and model parallelism.
- Synchronous distributed training can be done in a single where multiple GPUs are available using MirroredStrategy.
- On the other hand, MultiWorkerMirroredStrategy performs synchronous training on multiple workers or devices (either on GPUs or CPUs) 
- Federated Learning is a distributed machine learning approach that enables training models on decentralized data sources without the need for centralized data collection. It aims to address privacy concerns associated with sharing sensitive data while still allowing for model training and improvement.

- tf.GradientTape() records the operations involving tensors within its context, and when you call tape.gradient() with a target value and a list of variables, it uses the recorded operations to compute the gradients of the target with respect to the variables.
- Note that, tf.GradientTape() context and the tf.function-decorated function operate independently.

# Day 4
🎯 day 4 of #365daysofML
Keras is the high-level API of the TensorFlow platform. It provides an approachable, highly-productive interface for solving machine learning (ML) problems, with a focus on modern deep learning. Keras covers every step of the machine learning workflow, from data processing to hyperparameter tuning to deployment. It was developed with a focus on enabling fast experimentation.


✅ Daily logs
---
- learned about keras Layer and Model class
- learned different approaches to create models in keras which includes Sequential, function API and subclass method
- Understood upsampling and transpose convolution layers.

🔑 Keypoints

- keras.layers.Layers is inherited from the base class tf.Module and ketas.Model is inherited from the base class keras.layers.Layers
- Input layers is not mandatory in Sequential model, but if you want to view the model summary, then you should add it.
- A Sequential model is appropriate for a plain stack of layers where each layer has exactly one input tensor and one output tensor.
- Once a Sequential model has been built, it behaves like a Functional API model. This means that every layer has an input and output attribute.
- KerasTensors are used within the Keras API to define the inputs, outputs, and intermediate layers of a neural network model.
- While a regular TensorFlow tensor is a general-purpose tensor used in TensorFlow computations, 
  a KerasTensor is a specialized tensor object used within the Keras API, providing additional capabilities 
  such as automatic shape inference and automatic differentiation.
- The Keras functional API is a way to create models that are more flexible than the keras.Sequential API. The functional API can handle models with non-linear topology, shared layers, and even multiple inputs or outputs.
- Functional API can be used to use the same graph of layers to define multiple models.
- In function API, input layer is mandatory, however you can pass None for the input shape (at least last dim value should be provided i.e num of channal for CNN) to create dynamic input model.
- A model can contain sub-models (since a model is just like a layer). A common use case for model nesting is ensembling. For example, here's how to ensemble a set of models into a single model that averages their predictions
- The functional API makes it easy to manipulate multiple inputs and outputs. This cannot be handled with the Sequential API.
- Another good use for the functional API are models that use shared layers. Shared layers are layer instances that are reused multiple times in the same model -- they learn features that correspond to multiple paths in the graph-of-layers.

- All models in the keras API can interact with each other, whether they're Sequential models, functional models, or subclassed models that are written from scratch.

# day 5

🎯 day 5 of #365daysofML
loss is an optimization criterion used during training to adjust the model's parameters, while metrics are evaluation measures used to assess the model's performance on unseen data or in real-world scenarios.

📝 Message
I was quite busy today, thus, haven't got much time to get engaged in learning.

✅ Daily logs
---
- learned about different methods to create custom loss function in keras
   - Function based: This method involves creating a function that accepts inputs y_true and y_pred.  
   - Subclassing based: This method allows to pass parameters beside y_true and y_pred
   - Loss Function Wrapper: define loss function as a method, and use the LossFunctionWrapper to turn it into a class
   - Nasted function (not mentioned in the official documentation): Alternative to the Subclassing based custom loss function.

- learned about how to create custom metrics in keras
  - we can easily create custom metrics by subclassing the tf.keras.metrics.Metric class.

🌐 Resources
I am using official documentation of the tensorflow. In addition to that I will be using ChatGPT, medium articles, and stackoverflow to get in-depth insight.

🔗 Github Repo: https://github.com/s-4-m-a-n/365-days-of-ML-challenge/tree/main/keras%20basic%20to%20advance


# day 6 & 7
🎯 day 6 & 7 of #365daysofML

Binary logistic regression is a type of regression analysis used when the dependent variable (the outcome) is binary or dichotomous, meaning it can take only two possible values (e.g., yes/no, true/false, 0/1). Where as, Multiclass logistic regression, also known as softmax regression, is an extension of binary logistic regression to handle multiple classes (more than two) in the dependent variable.

📝 Message
I have decided to implement the logistic regression model to classify image using tensorflow core because of the two main reasons:
1. my ultimate goal is to implement computer vision and NLP research papers for which the core understanding of tensorflow and keras may needed
2. implementation of backpropagation using symbolic differentiation is not that efficient and is more complex task, specially when you are going to implement complex deeplearning architecture. 

During this implementation, I have encountered the vanishing and exploding gradient problems due to sigmoid activation function and weight initialization strategy respectively. I am going to make a saperate notebook to cover this scenario. 


✅ Daily logs
---
- have implemented multiclass logistic regression to classify MNIST dataset using tensorflow core.
- have implemented binary logistic regression to classify the digit 0 and 1 of the MNIST dataset using tensorflow core
- Understood and handled the vanishing and exploding problem

🔗 Github Repo: https://github.com/s-4-m-a-n/365-days-of-ML-challenge/tree/main/architecture%20implementations/Logistic%20Regression

# day 8
One important characteristic of the sigmoid function's derivative is that as the input (x) becomes very large or very small, the derivative approaches 0. 
This phenomenon is known as the "vanishing gradient" problem. It can hinder the learning process in deep neural networks, 
making it challenging for the network to update weights in earlier layers effectively.


📝 Message
Vanishing gradient problem is one of the problem we can face in deep neural networks. I have used two activation function i.e sigmoid and softmax in a multiclass logistic regression to cause a vanishing gradient problem. Note that in practice we will never use two activation function simultaneously in a same layer, I have done it to mimic (at some level) the deep neural network using logistic regression. I have no idea if it is mathematically making any sense or not, the main purpose is to analyze and understand the model and its outputs as well as the impact of weight initialization technique.


✅ Daily logs
---
- performed detailed analysis on vanishing gradient problem due to sigmoid function and weight initialization strategy

🔗 Github Repo: https://github.com/s-4-m-a-n/365-days-of-ML-challenge/blob/main/architecture%20implementations/Logistic%20Regression/Vanishing%20Gradient%20problem.ipynb


# day 11
Generative AI and Large Language Model (LLM) are specifically a general purpose technology, similar to other general purpose technologies like deeplearning and electricity.

Recommendation System is a system that predicts ratings or preferences that a user might give to an item. So, a recommender system is recommending things to people based on their past behavior and the behavior of other people.

📝 Message
During the previous week, I was engaged in travel, limiting my availability to delve into topics related to Machine Learning. In the present week, my objective is to successfully complete the "Generative AI with Large Language Models" course, available on Coursera through the collaborative efforts of deeplearning.ai and AWS. Simultaneously, I am immersing myself in the study of recommendation systems, as part of my ongoing learning endeavors.

✅ Daily logs
- completed the first week material of the generative AI with LLMs course
- Implementing CNN from scratch
- Covered some basic topic of recommendation system including evaluation metrics and challenges.

🔗 Github Repo: https://github.com/s-4-m-a-n/365-days-of-ML-challenge/tree/main/recommendation%20system


🔑 Keypoints
- Prompt is the input given to the LLM, where as completion is the output given by the model.
- The act of using model to generate text is known as inference.
- The space or memory that is available to the prompt is called context window.
- Prompt engineering is crafting a clear and specific question or instruction to get the desired information from a LLM. 
 It is about finding the right way to ask so that model understands what you want and provides a relivant answer.
- Providing examples inside the context window is called in-context learning. In zero-shot inference, no example is provided, in one-shot inference, one example is provided and in few-shot, few examples are provided.
- Variants of Transformer architecture includes:
   - **Encoder-only models (AutoEncoding model)**
      - they are pretrained using mask language modeling (MLM)
      - train objective: predict the masked token
      - Good usecases: sentiment analysis, NER, word classidication
      - Examples: BERT, ROBERTA
   - **Decoder-only models (AutoRegressive model)**
      - they are pretrained using Casual Language Modeling (CLM)
      - train objective: predict next token base on the previous sequence of tokens
      - Good usecases: text generation
      - examples: GPT, BLOOM
   - **Encoder-Decoder model (Sequence-to-sequence model)**:
      - they are pretrained using span corruption
         
         ``the | teacher | <mask\> | <mask\>| student``
         
         ``the | teacher |       <x\>        | student``
      - train objective: reconstruct span 
         
         ``<x\> | teaches | the |``
      - Good usecases: translation, Text summarization, question answering
      - Examples: T5, BART (not BERT) 


# day 12 & 13
Fine-tuning, in the context of machine learning, refers to the process of taking a pre-trained model and training it further on a specific task or dataset to improve its performance on that task. Fine-tuning is commonly used in transfer learning, where knowledge gained from solving one problem is applied to a related problem.

📝 Message
Did not get much part of the reinforcement learning, so will revise the week 3 once I cover fundamental topics of reinforcement learning.
I Will add keypoints tomorrow. 

✅ Daily logs
- completed the the generative AI with LLMs course


🔗 Github Repo: https://github.com/s-4-m-a-n/365-days-of-ML-challenge/tree/main/recommendation%20system



# Day 14
Its been a long since I hold this 365 days of ML challenge, so I decided to continue it from where I left. I am parallely doing 30 days of DevOPs challenge too 

📝 Message
- 

✅ Daily logs
- Today I have implemented language of a sentence classifier using character level simple RNN
- I have also implemented simpleRNN and DeepRNN from scratch using pytorch


# Day 15

✅ Daily logs
- Today I have implemented simple LSTM, DeepLSTM and Bidrectional LSTM from scratch using pytorch
- Haven't got time to document it, will do that in a holiday 


# Day 16

✅ Daily logs
- Today I have implemented simple GRU
- Also created a toy task to predict next number from the sequence to test the sanity of the implemented model 
