#ifndef NET_H
#define NET_H

#include <torch/torch.h>
#include <iostream>

struct NetImpl : torch::nn::Module
{
    NetImpl(int64_t m, int64_t n)
        : linear1(register_module("linear1", torch::nn::Linear(2, m))),
          linear2(register_module("linear2", torch::nn::Linear(m, n))),
          linear3(register_module("linear3", torch::nn::Linear(n, 1)))
    {}

    torch::Tensor forward(torch::Tensor x)
    {
//        cout << "x1: " << x << endl;
        x = torch::nn::functional::relu(linear1(x));
//        cout << "x2: " << x << endl;
        x = torch::nn::functional::relu(linear2(x));
//        cout << "x3: " << x << endl;
        x = linear3(x);
//        cout << "x4: " << x << endl;
        return x;
    }
    
    torch::nn::Linear linear1, linear2, linear3;
};
TORCH_MODULE(Net);

#endif
