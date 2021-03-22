# CUDACDNNs

I have a couple of goals with this repository. The most obvious is implementing Neural Networks using CUDA C. There are a number of repositories
that offer CUDA C++ implementations of Neural Networks but surprisingly few that offer C. I've never actually learned C++, so this will be in C!
I don't claim to be a great C programmer, or even a good one, that said for work I write almost exclusively in Python and so in my free time 
I really enjoy the simplicity that C offers, problem solving is how to better an algorithm as opposed to looking up documenation for a package.

One of the interesting part of all this is just how to structure the data together. Pytorch and TensorFlow in particular do an incredible job abstracting away things like gradients, backward passes and all that, where as implemented manually one has to find ways to keep track of all these pieces of data. In the case of C this becomes even more problematic as C is not an Object Oriented language, and does not have variable length arrays, so one has to both store the data in intuitive ways as well as find location to store said data. Honestly while this is written with a decent amount of CUDA, the CUDA is really an after thought with the aforementioned problems.

The first goal of this is to create a super simple linear network but in doing so start to create some of the objects needed to go forward in to more complex areas such as MNIST and maybe some day GANs.


My goal as of now is to complete the following:
1. Simple Dense Network: Given Matrix X and Vector Y train network to find yhat
2. Mnist Dense Network
3. Mnist CNN
4. CycleGAN
5. Time Series GAN
6. RL (something with an environment easy to implement in C, haha)
