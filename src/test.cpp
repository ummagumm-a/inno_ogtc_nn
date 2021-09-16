#include <iostream>
#include <fstream>
#include <vector>
#include <utility>
#include <torch/torch.h>

#include "Net.h"
#include "NetProcessing.h"
#include "DatasetModule.h"

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
        net_info >> tmp;
        vec.push_back(tmp);
    }

    return vec;
}

void save_data(vector<pair<double, double>> xy, vector<double> label, vector<double> output)
{
    ofstream data("../visualization/act_vs_pred.txt");

    if (data.is_open())
        for (int i = 0; i < xy.size(); ++i)
            data << xy[i].first << " "
                 << xy[i].second << " "
                 << label[i] << " "
                 << output[i] << endl;
    else
        cout << "Problems with opening file for saving test info." << endl;

    data.close();
}

int main()
{
    vector<int64_t> mn = load_net_info();
    Net net(mn[0], mn[1], mn[2]);
    net->to(torch::kFloat64);

    torch::load(net, "model.pt");
    cout << "Model is loaded" << endl;
 
    auto dataset = DatasetModule::MyDataset("../datasets/test.csv");
    dataset.normalize();
    auto data_loader = torch::data::make_data_loader(
            move(dataset),
            torch::data::DataLoaderOptions().batch_size(64));

    vector<pair<double, double>> xy_vec;
    vector<double> labels_vec;
    vector<double> output_vec;

    for (auto batch : *data_loader)
    {
        for (torch::data::Example<>& el : batch)
        {
            net->zero_grad();
            auto size = el.data.size(0);

            torch::Tensor output = net->forward(el.data.to(torch::kFloat64));

            xy_vec.push_back({ el.data[0].item<double>(), el.data[1].item<double>() });
            labels_vec.push_back(el.target.item<double>());
            output_vec.push_back(output.item<double>());

        }
    }

    save_data(xy_vec, labels_vec, output_vec);

    return 0;
}
