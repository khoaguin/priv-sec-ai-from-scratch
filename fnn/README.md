# Fully Connected Neural Network
This implementation is my solutions to the first homework of the course Deep Learning at NYU, 2021. Link: https://atcold.github.io/NYU-DLSP21/.  

`theory.pdf` contains my answers for the theory part in `homework1.pdf`. I try to make things as clear as possible, however, there are parts that I still have doubts. It is great if you can walk through them and find out my mistakes. Please make an issue to notify me about any mistakes, I am more than happy to discuss and fix them.  

The implementation part is in `mlp.py`. Based on the coure's homework, torch `Tensor` is used for convenience. Torch tensors are similar to numpy arrays, but tensors can be oprated on CUDA GPUs, and they can also keep the gradients of the loss function w.r.t themselves for automatic backward. Learn more about their differences [here](https://medium.com/@ashish.iitr2015/comparison-between-pytorch-tensor-and-numpy-array-de41e389c213). In this implementation, we will not use the torch's autograd but implement the backward pass ourselves.
