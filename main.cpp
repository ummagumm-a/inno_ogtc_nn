#include <iostream>
#include <vector>
#include <random>
#include <fstream>
#include <optional>
#include <torch/torch.h>

using namespace std;

double f(double x, double y)
{
    return x * y;
}

void gen_set(const string& path, int n)
{
    random_device dev;
    mt19937 rng(dev());
    uniform_real_distribution<> dist(-100, 100);

    ofstream data_file;
    data_file.open(path);
    if (data_file.is_open())
    {
        data_file << "x,y,f\n";

        for (int i = 0; i < n; ++i)
        {
            double x = dist(rng);
            double y = dist(rng);
            data_file << x << ","
                      << y << ","
                      << f(x,y) << "\n";
        }
    }

    data_file.close();
}

struct Net : torch::nn::Module
{
    Net(int64_t m)
        : linear1(torch::nn::Linear(2, m)),
          linear2(torch::nn::Linear(m, 1))
    {
        register_module("linear1", linear1);
        register_module("linear2", linear2);
    }

    torch::Tensor forward(torch::Tensor x)
    {
        cout << "1: " << x << endl;
        x = torch::relu(linear1(x));
        cout << "2: " << x << endl;
        x = linear2(x);
        cout << "3: " << x << endl;
        return torch::relu(x);
    }
    
    torch::nn::Linear linear1, linear2;
};

vector<string> split(const std::string& str, char delim)
{
    stringstream ss;
    ss << str;
    string seg;
    vector<string> seglist;

    while(getline(ss, seg, delim))
        seglist.push_back(seg);

    return seglist;
}

vector<vector<double>> read_data(const std::string& loc)
{
    ifstream infile;
    infile.open(loc);

    string line;
    vector<vector<double>> values;

    if (infile.is_open())
    {
        getline(infile, line);
        while (getline(infile, line))
        {
            vector<double> vec;
            for (const auto& el : split(line, ','))
                vec.push_back(stod(el));

            values.push_back(vec);
        }
    }
    else
        cout << "file is closed" << endl;
    
    infile.close();    

    return values;
}

class MyDataset : public torch::data::Dataset<MyDataset>
{
public:
    explicit MyDataset(const string& loc)
        : dset(read_data(loc)) 
    {}

    torch::data::Example<torch::Tensor, torch::Tensor> get(size_t index) override
    {
        torch::Tensor states, label;
        auto line = dset[index];
        vector<double> flag = { line[2] };
        line.pop_back();

        auto options = torch::TensorOptions().dtype(torch::kFloat64);
        states = torch::from_blob(line.data(), 
                {line.size()}, 
                options).clone();
        label = torch::from_blob(flag.data(), 
                {flag.size()}, 
                options).clone();

        return { states, label };
    }

    void normalize()
    {
        for (auto& s : dset) 
            normalize_line(s);
    }

    void unnormalize()
    {
        for (auto& s : dset) 
            unnormalize_line(s);
    }

    torch::optional<size_t> size() const override
    {
        return dset.size();
    }
private:
    void find_max(vector<vector<double>> vecs)
    {
        vector<double> max_vec;

        for (const auto& t : vecs)
            for (int i = 0; i < t.size(); ++i)
                max_vec[i] = max(max_vec[i], t[i]);

        maxes = max_vec;
    }

    void normalize_line(vector<double>& line)
    {
        for (int i = 0; i < maxes.size(); ++i)
            line[i] /= maxes[i];
    }
        
    void unnormalize_line(vector<double>& line)
    {
        for (int i = 0; i < maxes.size(); ++i)
            line[i] *= maxes[i];
    }

private: 
    vector<vector<double>> dset;
    vector<double> maxes;
};
  
auto create_dataset(const string& path)
{
    auto dataset = MyDataset(path)
        .map(torch::data::transforms::Normalize<>(0.5, 0.5))
        .map(torch::data::transforms::Stack<>());

    int64_t batch_size = 64;
    auto data_loader = torch::data::make_data_loader(
            move(dataset), 
            torch::data::DataLoaderOptions().batch_size(batch_size));

    return data_loader;
}

int main()
{
//    gen_set("../train.csv", 80000);
//    gen_set("../val.csv", 20000);

//    Net net(32);
//    for (const auto& p : net.parameters())
//        cout << p << endl;
   
    auto train_set = create_dataset("../train.csv");    
    auto val_set = create_dataset("../val.csv");    

    for (auto& batch : *val_set)
    {
        cout << "Batch size: " << batch.data.size(0) << " | Labels: ";
        for (int64_t i = 0; i < batch.data.size(0); ++i)
            cout << batch.target[i].item<double>() << " ";
        cout << endl;
    }

    return 0;
}
