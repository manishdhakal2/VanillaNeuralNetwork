import numpy as np

#Neural Network Implemented for MNIST DIGIT CLASSIFICATION

class NeuralNetwork:
    def __init__(self,learning_rate,epoch,neuron_no):
        self.learning_rate=learning_rate
        self.epoch=epoch
        self.neuron_no=neuron_no #no of neurons


    def ReLU(self,x):
        return np.maximum(0.01*x,x) #LeakyRelu : Alpha :0.01
    
    def softmax(self,z):
        exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))  
        return exp_z / np.sum(exp_z, axis=1, keepdims=True) # return the softmax probabilities

    def ReLU_gradient(self,x):
        """Returns the gradient of the RELU Function"""
        return np.where(x>0,1,0.01) 

    def one_hot_encoding(self,y,size):

        """One Hot Encodes each label """
        b=np.eye(size)[y]
        return b

    
    
    def train(self,x,y):

        """Initial training method
            x:np.array,y:np.array"""
        
        self.m,self.n=x.shape
        self.y=self.one_hot_encoding(y,10)
        self.init_weights()
        self.x=x
     
        for i in range(self.epoch):
            print("Iteration "+str(i))
            self.forward_prop()
            if i % 10 == 0:
                predictions = self.predict(self.x)
                accuracy = np.mean(np.argmax(predictions, axis=1) == y) * 100
                print(f"Training Accuracy: {accuracy:.2f}%")
            self.backward_prop()
    
    def init_weights(self):
        self.w1 = np.random.randn(self.n,self.neuron_no)*(np.sqrt(2./self.n))
        self.b1 = np.zeros(self.neuron_no)
        self.w2 = np.random.randn(self.neuron_no,self.neuron_no)*(np.sqrt(2./self.neuron_no))
        self.b2 = np.zeros(self.neuron_no)
        self.w3 = np.random.randn(self.neuron_no,10)*(np.sqrt(2./self.neuron_no))
        self.b3 = np.zeros(10)


    def forward_prop(self):
        """Forward Prop For Training"""
        
        self.z1 = np.dot(self.x, self.w1) + self.b1
        self.o1 = self.ReLU(self.z1)
        
        self.z2 = np.dot(self.o1, self.w2) + self.b2
        self.o2 = self.ReLU(self.z2)
        
        self.z3 = np.dot(self.o2, self.w3) + self.b3

        self.o3 = self.softmax(self.z3)
        
    def backward_prop(self):

        """Adjusts the weights and biases in each epoch"""
        
        error3=self.o3-self.y
        dw3=np.dot(self.o2.T,error3)/self.m
        db3=np.sum(error3,axis=0)/self.m
        self.w3-=self.learning_rate*dw3
        self.b3-=self.learning_rate*db3


        error2=np.dot(error3,self.w3.T)*self.ReLU_gradient(self.o2)
        dw2=np.dot(self.o1.T,error2)/self.m
        db2=np.sum(error2,axis=0)/self.m
        self.w2-=self.learning_rate*dw2
        self.b2-=self.learning_rate*db2

        error1=np.dot(error2,self.w2.T)*self.ReLU_gradient(self.o1)
        dw1=np.dot(self.x.T,error1)/self.m
        db1=np.sum(error1,axis=0)/self.m
        self.w1-=self.learning_rate*dw1
        self.b1-=self.learning_rate*db1

    
    
    def predict(self,x):
        """Inference Usage"""
        
        z1=np.dot(x,self.w1)+self.b1
        o1=self.ReLU(z1)
        z2=np.dot(o1,self.w2)+self.b2
        o2=self.ReLU(z2)
        z3=np.dot(o2,self.w3)+self.b3
        o3=self.softmax(z3)
        return o3
