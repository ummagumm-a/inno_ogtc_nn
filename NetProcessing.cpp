#include <torch/torch.h>

#include "NetProcessing.h"
#include "DatasetModule.h"

using namespace std;

double NetProcessing::test(Net& net, 
            const string& test_set_location)
{
    static auto test_set = DatasetModule::create_dataset(test_set_location);    

    int count = 0;
    double av_loss = 0;
    for (torch::data::Example<>& batch : *test_set)
    {
        net->zero_grad();
        torch::Tensor data = batch.data;
        torch::Tensor labels = batch.target;
        torch::Tensor output = net->forward(data.to(torch::kFloat64));

        torch::Tensor d_loss = torch::nn::functional::mse_loss(output, labels);
        av_loss += d_loss.item<double>();
        count++;
    }

    return av_loss / (double) count;
}

void NetProcessing::train(Net& net, 
           const string& train_set_location,
           int number_of_epochs)
{
    torch::optim::Adam net_optimizer(net->parameters(), torch::optim::AdamOptions(3e-5));
   
    auto train_set = DatasetModule::create_dataset(train_set_location);    
    
    for (int64_t epoch = 1; epoch <= number_of_epochs; ++epoch)
    {
        double av_loss = 0;
        int count = 0;

        for (torch::data::Example<>& batch : *train_set)
        {
            torch::Tensor data = batch.data;
            torch::Tensor labels = batch.target;
            torch::Tensor output = net->forward(data.to(torch::kFloat64));
            torch::Tensor d_loss = torch::nn::functional::mse_loss(output, labels);

            d_loss.backward();
            net_optimizer.step();

            av_loss += d_loss.item<double>();
            count++;
        }
        printf("train: %.07lf\ttest: %.07lf\n", av_loss / (double) count, test(net));
    }
}

