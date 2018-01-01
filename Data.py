import numpy as np

class Data:
    def __init__(self, echo_step, length_of_numbers):
        self.echo_step = echo_step
        self.length_of_numbers = length_of_numbers

    def get_data(self):
        return self._generateData()

    def _generateData(self):
        x = np.array(np.random.choice(2, self.length_of_numbers, p=[0.5, 0.5]))
        y = np.roll(x, self.echo_step)
        y[0:self.echo_step] = 0
        return (x, y)

if __name__ == "__main__":
    d = Data(3, 20)
    dat = d.getData()
    print dat[0]
    print dat[1]

