import numpy as np

class Linear:
    def __init__(self,in_features,out_features):
        self.in_features=in_features
        self.out_features=out_features
        self.w=None
        self.b=None
    
    def init_weights(self,inputShape):
        self.w=np.random.uniform(0,1,(self.out_features,inputShape))
        self.b=np.zeros(self.out_features)
    
    def forward(self,x):
        if not self.w:
            self.init_weights(x.shape[1])
        
        return np.dot(self.x,self.w)+self.b


    def backward(self,error):
        pass

        
