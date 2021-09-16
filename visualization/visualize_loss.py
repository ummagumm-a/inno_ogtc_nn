import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D

def read_loss_data(filename):
    loss = []
    with open(filename) as loss_data:
        loss_lines = loss_data.read().splitlines()
        loss = list(map(float, loss_lines))

    return loss
       
train_loss = read_loss_data('train_loss.txt')[2:]
val_loss = read_loss_data('val_loss.txt')[2:]
epochs = range(3, len(train_loss) + 3)

plt.plot(epochs, train_loss, 'b-', label='Train loss')
plt.plot(epochs, val_loss, 'r-', label='Validation loss')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.title('Train and validation loss during training')
plt.legend(loc="best")

plt.show()
