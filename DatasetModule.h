#ifndef DATASET_MODULE_H
#define DATASET_MODULE_H

#include <string>
#include <vector>
#include <fstream>
#include <torch/torch.h>

using namespace std;

class DatasetModule
{
public:
    class MyDataset : public torch::data::Dataset<MyDataset>
    {
    public:
        // to obtaion dataset, provide location of a csv file
        // where the data is stored
        explicit MyDataset(const string& loc)
            : dset(read_data(loc)) 
        {
            maxes = { 100, 100, 10000 };
        }

        // return one line from a dataset
        torch::data::Example<> get(size_t index) override;
        // normalize the whole dataset
        void normalize();
        // unnormalize the whole dataset
        void unnormalize();
        // returns the size of the dataset
        torch::optional<size_t> size() const override;

        void print_dataset() const;
    private:
        // normalize one line of a dataset
        void normalize_line(vector<double>& line);
        // unnormalize one line of a dataset
        void unnormalize_line(vector<double>& line);
    private: 
        // matrix of values in a dataset
        vector<vector<double>> dset;
        // vector of max value in each column
        vector<double> maxes;
    };
public:
    // split a string line by delimeter
    static vector<string> split(const std::string& str, char delim);

    // read data from a csv file, storing it into a matrix
    static vector<vector<double>> read_data(const std::string& loc);

    // create dataset and return dataloader
    static decltype(auto) create_dataset(const string& path, int batch_size)
    {
        auto tmp = MyDataset(path);
        tmp.normalize();

        auto dataset = tmp
            .map(torch::data::transforms::Stack<>());

        auto data_loader = torch::data::make_data_loader(
                move(dataset), 
                torch::data::DataLoaderOptions().batch_size(batch_size));

        return data_loader;
    }
};
 

#endif 
