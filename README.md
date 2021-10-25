# Cuda C Neural Networks

## Currenty under construction. 
## I'd recommend checking out my neural network library written in C which is working.

I have a couple of goals with this repository. The most obvious is implementing Neural Networks using CUDA C. There are a number of repositories
that offer CUDA C++ implementations of Neural Networks but surprisingly few that offer C. I've never actually learned C++, so this will be in C!
I don't claim to be a great C programmer, or even a good one, that said for work I write almost exclusively in Python and so in my free time 
I really enjoy the simplicity that C offers, problem solving is how to better an algorithm as opposed to looking up documenation for a package.

One of the interesting part of all this is just how to structure the data together. Pytorch and TensorFlow in particular do an incredible job abstracting away things like gradients, backward passes and all that, where as implemented manually one has to find ways to keep track of all these pieces of data. In the case of C this becomes even more problematic as C is not an Object Oriented language, and does not have variable length arrays, so one has to both store the data in intuitive ways as well as find location to store said data. Honestly while this is written with a decent amount of CUDA, the CUDA kernels really become an after thought with the aforementioned problems of just how to frame and store the data. 

Another interesting part are the results. The simple way to do this is to just printf the error and continue on, but ideally we would have something like Tensorboard where we get time series over time so we can view things like error, epsilons, validation runs etc. Along with this a way to save, export and import said neural networks. I saw another repository where the guy used Doxygen to report results which seemed like an interesting idea. 

The first goal of this is to create a super simple linear network but in doing so start to create some of the objects needed to go forward in to more complex areas such as MNIST and maybe some day GANs.


Also as a side note the computer I write the code on doesn't actually have an NVidia GPU so making this all a little more old school I probably won't be testing the code as often. 
