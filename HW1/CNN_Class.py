import numpy as np

class Network_Model:
    def __init__(self, loss, optimizer):
        self.layers = []
        self.loss = loss
        self.optimizer = optimizer
        
    def add(self, layer):
        self.layers.append(layer)
        
    def forward(self, data):
        for layer in self.layers:
            data = layer.forward(data)
        return softmax(data)
    
    def backward(self, grad, lr):
        for layer in reversed(self.layers):
            grad = layer.backward(grad, lr)
            print(grad)
    
    def train(self, train_set, test_set, epochs, lr, verbose):
        history = {'loss': [], 'val_loss': [], 'accuracy': [], 'val_accuracy': []}
        label_count = len(np.unique(test_set.targets))
        for epoch in range(epochs):
            if verbose:
                print("\n ==> Epoch {}".format(epoch))
            loss, tp = 0,0
            for i in range(len(train_set)):
                x, y = train_set[i]
                output = self.forward(np.asarray(x))
                loss += self.loss(self.layers, output[y])
                
                if np.argmax(output) == y:
                    tp += 1
                grad = np.zeros(label_count)
                grad[y] = -1 / output[y] + np.sum([2 * np.sum(np.absolute(layer.get_weights())) for layer in self.layers])
                
                lr = self.optimizer(lr, grad)
                
                self.backward(grad, lr)
            history["accuracy"].append(tp/len(train_set) * 100)
            history["loss"].append(loss / len(train_set))
            val_loss, val_accuracy = self.eval(test_set)
            history["val_loss"].append(val_loss)
            history["val_accuracy"].append(val_accuracy)
        if verbose:
            print("Train Loss: %02.3f" % (history["loss"][-1]))
            print("Train Accuracy: %02.3f" % (history["accuracy"][-1]))
            print("Validation Loss: %02.3f" % (history["val_loss"][-1]))
            print("Validation Accuracy: %02.3f" % (history["val_accuracy"][-1]))
    
    def eval(self, test_set):
        loss, tp = 0, 0
        for i in range(len(test_set)):
            output = self.forward(test_set[i][0])
            
            loss += self.loss(self.layers, output[test_set[i][1]])
            
            if np.argmax(output) == test_set[i][1]:
                tp += 1
            
        return loss / len(test_set), (tp / len(test_set)) * 100
            
def ReLU(x):
    return max(x, 0)
        
def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=0)

def cross_entropy(x):
    return -np.log(x)

def default_optimizer(lr, grad):
    return lr

def cross_entropy(layers, x):
    loss = -np.log(x)
    for layer in layers:
        loss += np.linalg.norm(layer.get_weights()) ** 2
    return loss

def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=0)

class Conv2D_Layer:
    def __init__(self, channels=8, stride=1, kernel_size=3, activation="relu"):
        self.channels = channels
        self.stride = stride
        self.kernel_size = kernel_size
        self.activation = activation
        self.filters = np.random.randn(channels, kernel_size, kernel_size) * 0.1
        self.last_shape = None
        self.last_input = None
    
    def forward(self, data):
        self.last_shape = data.shape
        self.last_input = data
        input_size = data.shape[1]
        output_size = int((input_size - self.kernel_size) / self.stride) + 1
        
        out = np.zeros((self.channels, output_size, output_size))
        
        for i in range(self.channels):
            for j in range(0, input_size - self.kernel_size + 1, self.stride):
                for k in range(0, input_size - self.kernel_size + 1, self.stride):
                    temp = data[:, j:j+self.kernel_size, k:k + self.kernel_size]
                    out[i, int(j/self.stride), int(k/self.stride)] += np.sum(self.filters[i] * temp)
        if self.activation == "relu":
            out = np.vectorize(ReLU)(out)
        return out
    
    def backward(self, grad, lr):
        input_size = self.last_shape[1]
            
        lossgrad_input = np.zeros(self.last_shape)
        lossgrad_filter = np.zeros(self.filters.shape)
        
        for i in range(self.channels):
            for j in range(0, input_size - self.kernel_size + 1, self.stride):
                for k in range(0, input_size - self.kernel_size + 1, self.stride):
                    temp = self.last_input[:, j:j+self.kernel_size, k:k + self.kernel_size]
                    lossgrad_filter[i] = np.sum(grad[i, int(j/self.stride), int(k/self.stride)] * temp, axis=0)
                    lossgrad_input[:, j:j+self.kernel_size, k:k+self.kernel_size] += grad[i, int(j/self.stride), int(k/self.stride)] * self.filters[i]
        
        self.filters -= lossgrad_filter * lr
        return lossgrad_input
    
    def get_weights(self):
        return np.reshape(self.filters, -1)
    
class MaxPooling2D_Layer:
    def __init__(self, size=2):
        self.size = size
        self.last_shape = None
        self.last_input = None
        
    def forward(self, data):
        self.last_shape = data.shape
        c, h, w = data.shape
        self.last_input = data
        out = np.zeros((c, h - self.size, w - self.size))
        
        for i in range(c):
            for j in range(h - self.size + 1):
                for k in range(w - self.size + 1):
                    temp = data[i, j:j+self.size, k:k + self.size]
                    out[i, j:j+self.size, k:k+self.size] = np.max(temp)
        
        return out
    
    def backward(self, grad, lr):
        c, h, w = self.last_shape
        out = np.zeros(self.last_shape)
        for i in range(c):
            for j in range(h - self.size):
                for k in range(h - self.size):
                    temp = self.last_input[i, j:j+self.size, k:k + self.size]
                    (x, y) = np.unravel_index(np.nanargmax(temp), temp.shape)
                    out[i, j:j+x, k:k+y] += grad[i, j, k]
        
        return out
    
    def get_weights(self):
        return 0
    
class FC_Layer:
    def __init__(self, innode, outnode, activation=None):
        self.innode = innode
        self.outnode = outnode
        self.biases = np.zeros(outnode)
        self.weights = np.random.randn(innode, outnode) * 0.1
        self.activation = activation
        self.last_shape = None
        self.last_input = None
        
    def forward(self, data):
        self.last_shape = data.shape
        
        data = data.flatten()
        out = np.dot(data, self.weights) + self.biases
        if self.activation == "relu":
            out = np.vectorize(ReLU)(out)
            
        self.last_input = data
        return out
    
    def backward(self, grad, lr):
        
        self.last_input = np.expand_dims(self.last_input, axis=1)
        grad = np.expand_dims(grad, axis=1)
        
        grad_weights = np.dot(self.last_input, np.transpose(grad))
        grad_biases = np.sum(grad, axis=1).reshape(self.biases.shape)
        
        self.weights -= grad_weights * lr
        self.biases -= grad_biases * lr
        
        out = np.dot(self.weights, grad).reshape(self.last_shape)
        return out
    
    def get_weights(self):
        return np.reshape(self.weights, -1)
        