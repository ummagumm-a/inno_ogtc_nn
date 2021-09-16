import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D
from operator import methodcaller

def read_loss_data(filename):
    xs = []
    ys = []
    zs = []
    output = []
    with open(filename) as data:
        data_lines = data.read().splitlines()
        data_str = list(map(methodcaller("split", " "), data_lines))
        data = [list(map(float, x)) for x in data_str]
        xs = [t[0] for t in data]
        ys = [t[1] for t in data]
        zs = [t[2] for t in data]
        output = [t[3] for t in data]

    return xs, ys, zs, output

xs, ys, zs, output = read_loss_data('act_vs_pred.txt')

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.scatter(xs, ys, zs, c="blue", label='Actual values', s=1)
ax.scatter(xs, ys, output, c="red", label='Predicted values', s=2)

ax.set_title('Actual vs. Predicted values for x,y')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
ax.legend(loc="best")

plt.show()
