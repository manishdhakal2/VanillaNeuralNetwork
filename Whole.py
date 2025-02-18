import numpy as np


class NeuralNetwork:
    def __init__(self,lr:float,neurons:int)->None:
        self.lr=lr
        self.neurons=neurons
        self.w=None
        self.b=None
    

    def init_weights(self,inputShape)->None:
        pass
        




    def forward(self,x:np.array)->np.array:
        if not self.w:
            self.init_weights(self.x.shape)

                





