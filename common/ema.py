



class EMA():
    def __init__(self, x, decay=0.99):
        self.decay = decay
        self.x = x
    
    def update(self, x):
        self.x = (1.0 - self.decay) * x + self.decay * self.x
        return self.x


