# PyES
Python library for training models with the Evolution Strategies (ES) algorithm.

Did you ever have a good idea for a loss function and realized that you cannot use it with gradient descent because it was not differentiable? Or did you ever just were not in the mood to calculate the derivative for your loss function? 
Do not despair because Evolution Strategies comes to the rescue!

The Evolution Strategies algorithm does not demand much of you, the only thing it wants of its loss (or fitness) function is that it can put predictions and target values in and gets some score out. 

If you are not convinced yet how cool this is, read [this](https://openai.com/blog/evolution-strategies/) blog article by OpenAI, where this implementation is based on.

This Repository consists of the actual algorithm and a dummy script as an example of how to use it. In this example, a neural network is trained on the MNIST dataset, using accuracy (which is not differentiable) as the fitness function!

TODO:
* Implement a baseline script with gradient descent and make a comparison
* Performance improvements
* Support for distributed training
