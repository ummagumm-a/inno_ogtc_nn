#include <iostream>
#include <torch/torch.h>

#include "Net.h"
#include "NetProcessing.h"

using namespace std;

int main()
{
    Net net(16, 16);
    net->to(torch::kFloat64);

    NetProcessing::train(net);

    return 0;
}
