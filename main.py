from Data import *
from Train import *

def main():
    echo_step = 2
    series_length = 50000
    num_epochs = 10
    batch_size = 5
    state_size = 4
    n_units = 15
    num_classes = 2

    d = Data(echo_step, series_length)
    t = Train(d, series_length, num_epochs, batch_size, state_size, n_units, num_classes)
    t.train()

if __name__ == "__main__":
    main()

