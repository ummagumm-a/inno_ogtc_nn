#ifndef NET_PROCESSING_H
#define NET_PROCESSING_H

#include <string>

#include "Net.h"

class NetProcessing
{
public:
    static double test(Net& net, 
            const std::string& test_set_location = "../datasets/val.csv");

    static void train(Net& net, 
            const std::string& train_set_location = "../datasets/train.csv",
            int number_of_epochs = 30);
};

#endif
