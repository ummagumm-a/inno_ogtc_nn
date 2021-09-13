#include <iostream>
#include <vector>
#include <random>
#include <torch/torch.h>

#include "DatasetModule.h"

using namespace std;

double f(double x, double y)
{
    return x * y;
}

void gen_set(const string& path, int n)
{
    random_device dev;
    mt19937 rng(dev());
    uniform_real_distribution<> dist(-100, 100);

    ofstream data_file;
    data_file.open(path);
    if (data_file.is_open())
    {
        data_file << "x,y,f\n";

        for (int i = 0; i < n; ++i)
        {
            double x = dist(rng);
            double y = dist(rng);
            data_file << x << ","
                      << y << ","
                      << f(x,y) << "\n";
        }
    }

    data_file.close();
}

struct Net : torch::nn::Module
{
    Net(int64_t m)
        : linear1(torch::nn::Linear(2, m)),
          linear2(torch::nn::Linear(m, 1))
    {
        register_module("linear1", linear1);
        register_module("linear2", linear2);
    }

    torch::Tensor forward(torch::Tensor x)
    {
        cout << "1: " << x << endl;
        x = torch::relu(linear1(x));
        cout << "2: " << x << endl;
        x = linear2(x);
        cout << "3: " << x << endl;
        return torch::relu(x);
    }
    
    torch::nn::Linear linear1, linear2;
};

int main()
{
//    gen_set("../train.csv", 80000);
//    gen_set("../val.csv", 20000);

//    Net net(32);
//    for (const auto& p : net.parameters())
//        cout << p << endl;
   
    auto train_set = DatasetModule::create_dataset("../train.csv");    
    auto val_set = DatasetModule::create_dataset("../val.csv");    

    for (auto& batch : *val_set)
    {
        cout << "Batch size: " << batch.data.size(0) << " | Labels: ";
        for (int64_t i = 0; i < batch.data.size(0); ++i)
            cout << batch.target[i].item<double>() << " ";
        cout << endl;
    }

    return 0;
}
