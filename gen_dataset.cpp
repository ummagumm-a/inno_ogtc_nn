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

int main()
{
    gen_set("../train.csv", 350000);
    gen_set("../val.csv", 150000);

    return 0;
}


