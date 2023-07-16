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
 
🔗  Github Repository: https://github.com/s-4-m-a-n/365-days-of-ML-challange/tree/main/tensorflow%20from%20basic%20to%20advance


#365daysofMLchallange
#machinelearning
#deeplearning
#MLOps
#tensorflow2
#keras


# Day 2:
🎯 day 2 of #365daysofML

🔥 Tensorflow from basic to advance:
---
TensorFlow is an open-source library developed by Google for numerical computation and machine learning tasks. It provides a comprehensive ecosystem of tools, libraries, and resources that facilitate the development and deployment of machine learning models.

🌐 Resources
---
I am using official documentation of the tensorflow.
In addition with the, I will be using ChatGPT, medium articles, and stackoverflow to get in-depth insight more precisely.

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
    - The persistence feature allows you to compute multiple gradients within the same tape context, but it doesn't retain information about changes to variables made after the initial gradient computation.
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


🔗  Github Repository: https://github.com/s-4-m-a-n/365-days-of-ML-challange/tree/main/tensorflow%20from%20basic%20to%20advance


#365daysofMLchallange
#machinelearning
#deeplearning
#MLOps
#tensorflow2
#keras


