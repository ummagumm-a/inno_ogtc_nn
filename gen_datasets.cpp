#include <iostream>
#include <fstream>
#include <vector>
#include <random>

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

int main(int argc, char **argv)
{
    if (argc == 3)
    {
        gen_set("../datasets/train.csv", stoi(argv[1]));
        gen_set("../datasets/val.csv", stoi(argv[2]));
    }

    return 0;
}


