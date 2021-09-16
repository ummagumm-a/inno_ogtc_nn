# inno_ogtc_nn
## Description
Implementation of neural network for f(x, y) surface interpolation.

## Dependencies
### C++
[libtorch](https://pytorch.org/cppdocs/installing.html)

cmake
### Python
matplotlib

## Build and Installation
mkdir build && cd build

cmake -DCMAKE_PREFIX_PATH=/path/to/libtorch ..

cmake --build . 

## How to use it
### Generate datasets: 
run __./gen_datasets__ *train_set_length* *validation_set_length* *test_set_length*
### Train the model:
run __./train__ *#nodes_first_layer* *#nodes_second_layer* *#nodes_second_layer* *learning_rate* *#epochs*
### Test the model:
run __./test__
### Test on your values:
run __./man_test__

## Visualization
First, you need to go to *visualization* directory.
### To visualize loss on each epoch
run *python3* *visualize_loss.py*
### To visualize actual values of f(x,y) and the prediction
Do it after calling ./test!

run *python3* *visualize_data.py*

## Example
### x * y
It worked well for me: __./train__ 64 32 16 0.00006 100
