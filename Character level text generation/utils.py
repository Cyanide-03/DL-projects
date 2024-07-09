import numpy as np

def softmax(z):
    exp=np.exp(z)
    return exp/np.sum(exp,axis=0)

def get_sample(sampled_indices,ix_to_char):
    word=''.join(ix_to_char[i] for i in sampled_indices)
    word=word[0].upper()+word[1:]
    return word

def initialize_parameters(n_a, n_x, n_y):
    Wax = np.random.randn(n_a, n_x)*0.01 # input to hidden
    Waa = np.random.randn(n_a, n_a)*0.01 # hidden to hidden
    Wya = np.random.randn(n_y, n_a)*0.01 # hidden to output
    ba = np.zeros((n_a, 1)) # hidden bias
    by = np.zeros((n_y, 1)) # output bias
    
    parameters = {"Wax": Wax, "Waa": Waa, "Wya": Wya, "ba": ba,"by": by}
    
    return parameters

def update_parameters(parameters, gradients, lr):
    parameters['Wax'] += -lr * gradients['dWax']
    parameters['Waa'] += -lr * gradients['dWaa']
    parameters['Wya'] += -lr * gradients['dWya']
    parameters['ba']  += -lr * gradients['dba']
    parameters['by']  += -lr * gradients['dby']
    return parameters

def rnn_step_forward(parameters,a_prev,x):
    Waa, Wax, Wya, by, ba = parameters['Waa'], parameters['Wax'], parameters['Wya'], parameters['by'], parameters['ba']
    a_next=np.tanh(np.dot(Wax,x)+np.dot(Waa,a_prev)+ba)
    y=softmax(np.dot(Wya,a_next)+by)
    
    return a_next,y

def rnn_forward(parameters,X,Y,a_prev,vocab_size=27):
    T_x=len(X)
    ba=parameters['ba']
    
    x=np.zeros((vocab_size,T_x))
    
    a=np.zeros((ba.shape[0],T_x+1))
    a[:, 0] = a_prev[:, 0]
    
    y_hat=np.zeros((vocab_size,T_x))

    loss=0
    for t in range(len(X)):
        if (X[t] != None):
            x[X[t],t] = 1
        
        a_next, y_hat_next = rnn_step_forward(parameters, a[:, t].reshape(-1, 1), x[:, t].reshape(-1, 1))
        
        a[:, t + 1] = a_next[:, 0]
        y_hat[:, t] = y_hat_next[:, 0]
            
        loss-=np.log(y_hat[Y[t],t])
    
    cache=(y_hat,a,x)
    return loss,cache

def rnn_step_backward(dy_hat,parameters,gradients,x,a_t,a_prevt):
    gradients['dWya']+=np.dot(dy_hat,a_t.T)
    gradients['dby']+=dy_hat
    
    da=np.dot(parameters['Wya'].T,dy_hat)+gradients['da_t']
    daraw=(1-a_t*a_t)*da
    gradients['dWaa']+=np.dot(daraw,a_prevt.T)
    gradients['dWax']+=np.dot(daraw,x.T)
    gradients['dba']+=daraw
    gradients['da_t']=np.dot(parameters['Waa'],daraw)
    
    return gradients

def rnn_backward(X,Y,parameters,cache):
    gradients={}
    T_x=len(X)
    (y_hat,a,x)=cache
    
    Waa, Wax, Wya, by, ba = parameters['Waa'], parameters['Wax'], parameters['Wya'], parameters['by'], parameters['ba']
    
    gradients['dWax']=np.zeros_like(Wax)
    gradients['dWya']=np.zeros_like(Wya)
    gradients['dWaa']=np.zeros_like(Waa)
    gradients['dba']=np.zeros_like(ba)
    gradients['dby']=np.zeros_like(by)
    gradients['da_t']=np.zeros((a.shape[0],1))
    
    for t in reversed(range(T_x)):
        dy_hat = np.copy(y_hat[:,t])
        dy_hat[Y[t]] -= 1              
        gradients=rnn_step_backward(dy_hat.reshape(-1,1),parameters,gradients,x[:,t].reshape(-1,1),a[:,t+1].reshape(-1,1),a[:,t].reshape(-1,1))

    return gradients,a
