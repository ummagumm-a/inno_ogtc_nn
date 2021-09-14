#include "DatasetModule.h"

vector<string> DatasetModule::split(const std::string& str, char delim)
{
    stringstream ss;
    ss << str;
    string seg;
    vector<string> seglist;

    while(getline(ss, seg, delim))
        seglist.push_back(seg);

    return seglist;
}

vector<vector<double>> DatasetModule::read_data(const std::string& loc)
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

torch::data::Example<> DatasetModule::MyDataset::get(size_t index) 
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

void DatasetModule::MyDataset::normalize()
{
    for (auto& s : dset) 
        normalize_line(s);
}

void DatasetModule::MyDataset::unnormalize()
{
    for (auto& s : dset) 
        unnormalize_line(s);
}

torch::optional<size_t> DatasetModule::MyDataset::size() const 
{
    return dset.size();
}

void DatasetModule::MyDataset::normalize_line(vector<double>& line)
{
    for (int i = 0; i < line.size(); ++i)
        line[i] /= maxes[i];
}
    
void DatasetModule::MyDataset::unnormalize_line(vector<double>& line)
{
    for (int i = 0; i < line.size(); ++i)
        line[i] *= maxes[i];
}
 
void DatasetModule::MyDataset::print_dataset() const
{
    for (int i = 0; i < 10; ++i)
    {
        for (const auto& pp : dset[i])
            cout << pp << " ";
        cout << endl;
    }
}
