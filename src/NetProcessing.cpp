#include <torch/torch.h>
#include <iostream>

#include "NetProcessing.h"
#include "DatasetModule.h"

using namespace std;

double NetProcessing::validate(Net& net, 
            const string& val_set_location)
{
    static auto val_set = DatasetModule::create_dataset(val_set_location);    

    int count = 0;
    double av_loss = 0;
    for (torch::data::Example<>& batch : *val_set)
    {
        net->zero_grad();
        auto size = batch.data.size(0);
        torch::Tensor data = batch.data;
        torch::Tensor labels = batch.target;
        torch::Tensor output = net->forward(data.to(torch::kFloat64));

        av_loss += torch::nn::functional::mse_loss(output, labels).item<double>();
        count++;
    }

    return av_loss / (double) count;
}

void NetProcessing::train(Net& net, 
           double learning_rate,
           const string& train_set_location,
           int number_of_epochs)
{
    torch::optim::Adam net_optimizer(net->parameters(), torch::optim::AdamOptions(learning_rate));
   
    auto train_set = DatasetModule::create_dataset(train_set_location);    

    vector<double> train_loss_log;
    vector<double> val_loss_log;
    
    for (int64_t epoch = 1; epoch <= number_of_epochs; ++epoch)
    {
        double av_loss = 0;
        int count = 0;

        for (torch::data::Example<>& batch : *train_set)
        {
            net->zero_grad();
            torch::Tensor data = batch.data;
            torch::Tensor labels = batch.target;
            torch::Tensor output = net->forward(data.to(torch::kFloat64));
            torch::Tensor d_loss = torch::nn::functional::mse_loss(output, labels);

            d_loss.backward();
            net_optimizer.step();

            av_loss += d_loss.item<double>();
            count++;
        }
        double train_loss = av_loss / (double) count;
        double val_loss = validate(net);

        // print result on this epoch
        printf("epoch [%d/%d]\ttrain loss: %.07lf\tvalidation loss: %.07lf\n", 
                epoch,
                number_of_epochs, 
                train_loss,
                val_loss);

        // save info about loss on this epoch
        train_loss_log.push_back(train_loss);
        val_loss_log.push_back(val_loss);
    }

    // save all info about loss while training and validating
    save_loss_data(train_loss_log, "train_loss.txt");
    save_loss_data(val_loss_log, "val_loss.txt");
}

double NetProcessing::use(Net& net, vector<double>& data)
{
    data = { data[0] / 100, data[1] / 100 };

    auto options = torch::TensorOptions().dtype(torch::kFloat64);
    torch::Tensor t_data = torch::from_blob(data.data(), 
            {data.size()}, 
            options).clone();


    net->zero_grad();
    return 10000 * net->forward(t_data.to(torch::kFloat64)).item<double>();
}

void NetProcessing::save_loss_data(vector<double> data, string dest)
{
    ofstream loss_data("../visualization/" + dest);

    if (loss_data.is_open())
        for (int i = 0; i < data.size(); ++i)
            loss_data << data[i] << endl;
    else
        cout << "Problems with loss_data" << endl;

    loss_data.close();
}
