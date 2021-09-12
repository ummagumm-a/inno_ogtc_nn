#include <iostream>
#include <vector>
#include <random>
#include <fstream>

using namespace std;

double f(double x, double y)
{
    return x * y;
}

void create_data_set()
{
    random_device dev;
    mt19937 rng(dev());
    uniform_real_distribution<> dist(-100, 100);


    ofstream data_file;
    data_file.open("set.csv");
    if (data_file.is_open())
    {
        data_file << "x,y,f\n";

        for (int i = 0; i < 1000; ++i)
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
//    create_data_set();

    return 0;
}
