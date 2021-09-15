#include <iostream>
#include <fstream>
#include <torch/torch.h>

#include "Net.h"
#include "NetProcessing.h"

using namespace std;

void save_net_info(int64_t m, int64_t n, int64_t k)
{
    ofstream net_info("net_info.txt");
    if (net_info.is_open())
        net_info << m << " " << n << " " << k << endl;

    net_info.close();
}

int main(int argc, char** argv)
{
    int64_t m, n, k;
    double lr;
    int n_epochs;

    if (argc == 6)
    {
        m = stoi(argv[1]);
        n = stoi(argv[2]);
        k = stoi(argv[3]);
        lr = stod(argv[4]);
        n_epochs = stoi(argv[5]);
    }
    else
    {
        m = 32;
        n = 16;
        k = 16;
        lr = 5e-5;
        n_epochs = 100;
    }

    Net net(m, n, k);
    net->to(torch::kFloat64);
    cout << "Model is initialized" << endl;

    NetProcessing::train(net, lr, "../datasets/train.csv", n_epochs);
    cout << "Model is trained" << endl;
    torch::save(net, "model.pt");
    save_net_info(m, n);
    cout << "Model is saved." << endl;

    return 0;
}

