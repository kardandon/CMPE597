# İzzet Emre Küçükkaya
# 2017401123
# CMPE597 HW1
import numpy as np
import numba
EPS = 0
# Network Class
class Network_Model:
    def __init__(self, loss, lr_scheduler):
        self.layers = []
        self.loss = loss
        self.lr_scheduler = lr_scheduler
        if loss == cross_entropy:
            self.dloss = dcross_entropy
        
    def add(self, layer):
        self.layers.append(layer)
        
    def forward(self, data):
        for layer in self.layers:
            data = layer.forward(data)
        return data
    
    def backward(self, grad, lr):
        for layer in reversed(self.layers):
            grad = layer.backward(grad, lr)
            
    #Train using dataloader 
    def train_set(self, train_set, test_set, epochs, lr, verbose):
        history = {'loss': [], 'val_loss': [], 'accuracy': [], 'val_accuracy': []}
        try:
            label_count = len(np.unique(test_set.targets))
        except:
            label_count = len(np.unique(test_set[1]))
        for epoch in range(epochs):
            loss, tp = 0,0
            if verbose:
                print("\n ==> Epoch {}".format(epoch))
                
            
            for i in range(len(train_set)):
                if verbose and i % 10 == 9:
                    print("accuracy: {} loss: {} iter:{} ".format(tp/i, loss/i, i), end="\r")
                x, y = train_set[i]
                output = self.forward(np.array(x))
                loss += self.loss(y, output)
                if np.argmax(output) == y:
                    tp += 1
                grad = self.dloss(y, output)

                lr = self.lr_scheduler(lr, i)

                self.backward(grad, lr)
            history["accuracy"].append(tp/len(train_set) * 100)
            history["loss"].append(loss / len(train_set))
            val_loss, val_accuracy = self.eval(test_set)
            history["val_loss"].append(val_loss)
            history["val_accuracy"].append(val_accuracy)
            if verbose:
                print("loss: {} accuracy: {} val_loss: {} val_accuracy: {} ".format(history["loss"][-1], history["accuracy"][-1], history["val_loss"][-1], history["val_accuracy"][-1]))
                
    # Train using seperate numpy arrays
    def train(self, X_train, Y_train, X_test, Y_test, epochs, lr, verbose):
        history = {'loss': [], 'val_loss': [], 'accuracy': [], 'val_accuracy': []}
        label_count = len(np.unique(Y_train))
        
        for epoch in range(epochs):
            loss, tp = 0,0
            if verbose:
                print("\n ==> Epoch {}".format(epoch))
                
            # calculate loss and backpropagate
            for i in range(len(Y_train)):
                if verbose and i % 10 == 9:
                    print("accuracy: {} loss: {} iter:{} ".format(tp/i, loss/i, i), end="\r")
                x = X_train[i]
                y = Y_train[i]
                output = self.forward(np.array(x))
                loss += self.loss(y, output)
                if np.argmax(output) == y:
                    tp += 1
                grad = self.dloss(y, output)

                lr = self.lr_scheduler(lr, i)

                self.backward(grad, lr)
            history["accuracy"].append(tp/len(Y_train) * 100)
            history["loss"].append(loss / len(Y_train))
            val_loss, val_accuracy = self.eval(X_test, Y_test)
            history["val_loss"].append(val_loss)
            history["val_accuracy"].append(val_accuracy)
            if verbose:
                print("loss: {} accuracy: {} val_loss: {} val_accuracy: {} ".format(history["loss"][-1], history["accuracy"][-1], history["val_loss"][-1], history["val_accuracy"][-1]))
    
    # Eval using dataloader
    def eval_set(self, test_set):
        loss, tp = 0, 0
        for i in range(len(test_set)):
            output = self.forward(np.asarray(test_set[i][0]))
            
            loss += self.loss([test_set[i][1]], output)
            
            if np.argmax(output) == test_set[i][1]:
                tp += 1
            
        return loss / len(test_set), (tp / len(test_set)) * 100
    
    # Eval using nump arrays
    def eval(self, X_test, Y_test):
        loss, tp = 0, 0
        for i in range(len(Y_test)):
            output = self.forward(np.asarray(X_test[i]))
            
            loss += self.loss(Y_test[i], output)
            
            if np.argmax(output) == Y_test[i]:
                tp += 1
            print("{} of {} accuracy: {}".format(i+1,len(Y_test), tp/(i+1)), end="\r")
        return loss / len(Y_test), (tp / len(Y_test)) * 100
    
    # return predicted labels
    def predict(self,X_test):
        pred_label = np.zeros(len(X_test))
        for i in range(len(X_test)):
            output = self.forward(X_test[i])
            pred_label[i] = np.nanargmax(output)
        return pred_label
     
# loss explodes sometimes thus i decreased lr as iteration increases
def lr_scheduler(learning_rate, iteration):
    if (iteration + 1) % 10000 == 0:
        return learning_rate * 0.1
    else:
        return learning_rate
    
def ReLU(x):
    return np.max(x, 0)
        
def dReLU(x):
    return 1 if x > 0 else 0

def softmax(x): 
    # exp overloads when it is large thus I used stable softmax when it happens
    out = (np.exp(x) / np.sum(np.exp(x), axis=0))
    if np.isnan(out).any():
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum(axis=0)
    return out

def dsoftmax(x):
    soft = softmax(x)
    return soft * (1-soft)

def default_lr(lr, iteration):
    return lr

def cross_entropy(y, y_pred):
    loss = -np.log(y_pred[y] + EPS)
    return loss#/float(y_pred.shape[0])

def dcross_entropy(y, y_pred):
    #dloss = np.zeros(y_pred.shape)
    #dloss[y] = - 1/ (y_pred[y] + EPS)
    #return dloss#/float(y_pred.shape[0]))
    y_pred[y] -= 1
    return y_pred

# parallelizing the convolutions
@numba.njit(parallel=True)
def conv2d(filters, data, channels, stride, kernel_size, input_size):
    output_size = int((input_size - kernel_size) / stride) + 1
    out = np.zeros((channels, output_size, output_size))
    # basic convolution
    for i in numba.prange(channels):
        for j in range(0, input_size - kernel_size + 1, stride):
            for k in range(0, input_size - kernel_size + 1, stride):
                for l in numba.prange(data.shape[0]):
                    temp = data[l, j:j + kernel_size, k:k + kernel_size]
                    out[i, int(j/stride), int(k/stride)] += np.sum(filters[i,:,:] * temp)
    return out

# parallelizing the convolutions
@numba.njit(parallel=True)
def conv2d_back(filters, grad, channels, stride, kernel_size, input_size, last_shape, last_input):
    lossgrad_input = np.zeros(last_shape)
    lossgrad_filter = np.zeros(filters.shape)
    # backpropagation explained in report
    for i in numba.prange(channels):
        for j in range(0, input_size - kernel_size + 1, stride):
            for k in range(0, input_size - kernel_size + 1, stride):
                for l in numba.prange(last_input.shape[0]):
                    temp = last_input[l, j:j+kernel_size, k:k + kernel_size]
                    lossgrad_filter[i, :, :] += grad[i, int(j/stride), int(k/stride)] * temp
                    lossgrad_input[l, j:j+kernel_size, k:k+kernel_size] += grad[i, int(j/stride), int(k/stride)] * filters[i, :, :]
    return lossgrad_input, lossgrad_filter
                
class Conv2D_Layer:
    def __init__(self, channels=8, stride=1, kernel_size=3, activation="relu", name="Conv2D", filters=None):
        self.name = name
        self.channels = channels
        self.stride = stride
        self.kernel_size = kernel_size
        self.activation = activation
        try:
            if filters == None:
                self.filters = np.random.randn(channels, kernel_size, kernel_size) * 0.1
        except:
            self.filters = filters
        self.last_shape = None
        self.last_input = None
        self.last_output = None
    
    def forward(self, data):
        self.last_shape = data.shape
        self.last_input = data
        input_size = data.shape[1]
       
        out = conv2d(self.filters, data, self.channels, self.stride, self.kernel_size, input_size)
        self.last_output = out
        # activation function
        if self.activation == "relu":
            out = np.vectorize(ReLU)(out)
        return out
    
    def backward(self, grad, lr):
        if self.activation == "relu": 
            last_output = np.vectorize(dReLU)(self.last_output)
            grad = last_output * grad
        
        input_size = self.last_shape[1]
        
        lossgrad_input, lossgrad_filter = conv2d_back(self.filters, grad, self.channels, self.stride, self.kernel_size, input_size, self.last_shape, self.last_input)
        #update weights
        self.filters -= lossgrad_filter * lr
        return lossgrad_input
    
    def get_weights(self):
        return self.filters, None
    
class MaxPooling2D_Layer:
    def __init__(self, size=2, name="Maxpool"):
        self.name = name
        self.size = size
        self.stride = size
        self.last_shape = None
        self.last_input = None
        
    def forward(self, data):
        self.last_shape = data.shape
        c, h, w = data.shape
        self.last_input = data
        output_size = int((h - self.size) / self.stride) + 1
        out = np.zeros((c, output_size, output_size))
        # max pooling
        for i in range(c):
            for j in range(0, h - self.size + 1, self.stride):
                for k in range(0, w - self.size + 1, self.stride):
                    xx = int(j / self.stride)
                    yy = int(k / self.stride)
                    temp = data[i, j:j+self.size, k:k + self.size]
                    out[i, xx, yy] = np.max(temp)
        return out
    
    def backward(self, grad, lr):
        c, h, w = self.last_shape
        out = np.zeros(self.last_shape)
        # putting gradient where was max
        for i in range(c):
            for j in range(0, h - self.size + 1, self.stride):
                for k in range(0, w - self.size + 1, self.stride):
                    xx = int(j / self.stride)
                    yy = int(k / self.stride)
                    temp = self.last_input[i, j:j+self.size, k:k + self.size]
                    (x, y) = np.unravel_index(np.nanargmax(temp), temp.shape)
                    out[i, j+x, k+y] += grad[i, xx, yy]
        
        return out
    
    def get_weights(self):
        return None, None
    
class FC_Layer:
    def __init__(self, innode, outnode, activation=None, name="FC", weights=None, biases=None):
        self.name = name
        self.innode = innode
        self.outnode = outnode
        try:
            if biases == None:
                self.biases = np.zeros(outnode)
        except:
            self.biases = biases
        try:
            if weights == None:
                self.weights = np.random.randn(innode, outnode) * 0.1
        except:
            self.weights = weights
        self.activation = activation
        self.last_shape = None
        self.last_input = None
        self.last_output = None
        
    def forward(self, data):
        self.last_shape = data.shape
        #flatten to make operations
        data = data.flatten()
        out = np.dot(data, self.weights) + self.biases
        
        if self.activation == "relu":
            out = np.vectorize(ReLU)(out)
        
        self.last_output = out 
        self.last_input = data
        return out
    
    def backward(self, grad, lr):
        last_out = self.last_output
        if self.activation == "relu":
            last_out = np.vectorize(dReLU)(last_out)
            grad = last_out * grad
        # Update weights
        grad_weights = np.dot(self.last_input[np.newaxis].transpose(), grad[np.newaxis])
        grad_biases = grad
        out = np.dot(self.weights, grad).reshape(self.last_shape)
        self.weights -= grad_weights * lr
        self.biases -= grad_biases * lr
        
        return out
    
    def get_weights(self):
        return self.weights, self.biases

class Softmax_Layer:
    def __init__(self, name="softmax"):
        self.name = name
        self.last_input = None
    
    def forward(self, data):
        self.last_input = data
        return softmax(data)
    
    def backward(self, grad, lr):
        return grad * dsoftmax(self.last_input)
    def get_weights(self):
        return None, None