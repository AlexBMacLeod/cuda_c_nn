# CUDACDNNs

I have a couple of goals with this repository. The most obvious is implementing Neural Networks using CUDA C. There are a number of repositories
that offer CUDA C++ implementations of Neural Networks but surprisingly few that offer C. I've never actually learned C++, so this will be in C!
I don't claim to be a great C programmer, or even a good one, that said for work I write almost exclusively in Python and so in my free time 
I really enjoy the simplicity that C offers, problem solving is how to better an algorithm as opposed to looking up documenation for a package.

With C, of course one has to implement all of the math, but one is also presented with other interesting problems such as how to best store data. C is
of course notoriously picky when it comes to memory and allowing the user to choose the number of layers as opposed to hardcoding them adds an interesting
new dimension to the problem of how to store that data, and also how to record weights and reload them. 

My goal as of now is to complete the following:
1. Simple Dense Network: Given Matrix X and Vector Y train network to find yhat
2. Mnist Dense Network
3. Mnist CNN
4. CycleGAN
5. Time Series GAN
6. RL (something with an environment easy to implement in C, haha)
