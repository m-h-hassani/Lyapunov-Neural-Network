# Lyapunov-Neural-Network
Here I will develop a neural network based on Lyapunov theory for controlling industrial plants.

This system is based on the following paper:
- Aftab, Muhammad Saleheen, and Muhammad Shafiq. "Adaptive PID controller based on Lyapunov function neural network for time delay temperature control." 2015 IEEE 8th GCC Conference & Exhibition. IEEE, 2015.

lypid.m is a control system based on the Lyapunov theory. There are 3 controlling blocks that each one is responsible for one of the PID parameters, Kp, Ki, and Kd.
- Activation function is tanh(.).
- Neural Network weights are initialized randomly.
- Instead of using d(k) = Ue(k) like the mentioned paper, I used d(k) = Ue(k-1) for NNE networks.



