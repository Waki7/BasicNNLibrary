

class Optimizer():
    def __init__(self, lr):
        self.lr = lr

    def Update(self, partialDerivative):
        raise NotImplementedError


class GD(Optimizer):
    def __init__(self, lr):
        super().__init__(lr)

    def Update(self, partialDerivative):
        return -self.lr * partialDerivative