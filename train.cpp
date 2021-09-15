#include <iostream>
#include <torch/torch.h>

#include "Net.h"
#include "NetProcessing.h"

using namespace std;

int main(int argc, char** argv)
{
    int64_t m, n;
    double lr;
    int n_epochs;

    if (argc == 5)
    {
        m = stoi(argv[1]);
        n = stoi(argv[2]);
        lr = stod(argv[3]);
        n_epochs = stoi(argv[4]);
    }
    else
    {
        m = 32;
        n = 16;
        lr = 5e-5;
        n_epochs = 100;
    }

    Net net(m, n);
    net->to(torch::kFloat64);
    cout << "Model is initialized" << endl;

    NetProcessing::train(net, lr, "../datasets/train.csv", n_epochs);
    cout << "Model is trained" << endl;
    torch::save(net, "../model.pt");
    cout << "Model is saved." << endl;

    return 0;
}

