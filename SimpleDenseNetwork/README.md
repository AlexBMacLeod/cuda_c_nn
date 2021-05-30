Why CUDA C? There was a point in time when I'd say because I know C and not C++, but making the classes that pull all this together in C++ is pretty straight forward and has a good deal of documentation. But the architecture of it all in C is pretty interesting, especially since all the others tend to be so object oriented. That said as you can see from my code I used a couple of typedefs in a way remiscent of OOP. So the question of how to store the data and make the interface becomes realy interesting.

The aim of this is to a simple feedforward neural network, the number of layers and neurons per layer can be set as args,
then the number of iterations, learning rate and all of that are adjustable hyperparameters in nn.cu.
