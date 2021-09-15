#include <iostream>
#include <fstream>
#include <vector>
#include <torch/torch.h>

#include "Net.h"
#include "NetProcessing.h"

using namespace std;

vector<int64_t> load_net_info()
{
    ifstream net_info("net_info.txt");
    vector<int64_t> vec;
    int64_t tmp;
    if (net_info.is_open())
    {
        net_info >> tmp;
        vec.push_back(tmp);
        net_info >> tmp;
        vec.push_back(tmp);
    }

    return vec;
}

int main()
{
    vector<int64_t> mn = load_net_info();
    Net net(mn[0], mn[1]);
    net->to(torch::kFloat64);

    torch::load(net, "model.pt");
    cout << "Model is loaded" << endl;
 
    cout << endl << "Print number of examples that you want to check" << endl;
    int n;
    cin >> n;

    double x, y;
    vector<double> inp_vec;
    cout << "Print each example in the format <x y>" << endl;
    for (int i = 0; i < n; ++i)
    {
        cin >> x >> y;
        inp_vec = { x, y };
        cout << "Answer: " << NetProcessing::use(net, inp_vec) << endl;
    }

    return 0;
}
