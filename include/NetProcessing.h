#ifndef NET_PROCESSING_H
#define NET_PROCESSING_H

#include <string>
#include <vector>
#include <fstream>

#include "Net.h"

class NetProcessing
{
public:
    struct Data
    {
        double x;
        double y;
        double label;
        double prediction;
    };

    static double validate(Net& net, 
            const std::string& val_set_location = "../datasets/val.csv");

    static void train(Net& net, 
            double learning_rate = 1e-3,
            const std::string& train_set_location = "../datasets/train.csv",
            int number_of_epochs = 30);

    static double use(Net& net, std::vector<double>& data);

    static void save_loss_data(std::vector<double>, std::string dest);
};

#endif
