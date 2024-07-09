import numpy as np

def softmax(z):
    expo=np.exp(z)
    return expo/np.sum(expo,axis=0)

def get_sampled_indices(sampled_indices,ix_to_char):
    word=''.join(ix_to_char[c] for c in sampled_indices)
    word=word[0].upper()+word[1:]
    return word

def initialize(n_x,n_a,n_y):
    parameters={}
    
    Waa=np.random.randn(n_a,n_a)*0.01
    Wax=np.random.randn(n_a,n_x)*0.01
    Wya=np.random.randn(n_y,n_a)*0.01
    ba=np.zeros((n_a,1))
    by=np.zeros((n_y,1))
    
    parameters['Waa']=Waa
    parameters['Wax']=Wax
    parameters['Wya']=Wya
    parameters['ba']=ba
    parameters['by']=by
    
    return parameters

def update_parameters(parameters,gradients,lr):
#     print("Gradients before update:")
#     for key, value in gradients.items():
#         print(f"{key}: {value}")
    
    parameters['Wax']+= -lr * gradients['dWax']
    parameters['Waa']+= -lr * gradients['dWaa']
    parameters['Wya']+= -lr * gradients['dWya']
    parameters['ba'] += -lr * gradients['dba']
    parameters['by'] += -lr * gradients['dby']
    return parameters
    

def rnn_step_forward(x_t,a_prev,parameters): # FOR SINGLE TIME STEP OF AN EXAMPLE
    Waa=parameters['Waa']
    Wax=parameters['Wax']
    Wya=parameters['Wya']
    ba=parameters['ba']
    by=parameters['by']
    
    a_t=np.tanh(np.dot(Wax,x_t)+np.dot(Waa,a_prev)+ba)
    y_t=softmax(np.dot(Wya,a_t)+by)
    
    return a_t,y_t

def rnn_forward(X,Y,parameters,a_prev): # FOR ALL TIME STEPS OF AN EXAMPLE
    Waa=parameters['Waa']
    Wax=parameters['Wax']
    Wya=parameters['Wya']
    ba=parameters['ba']
    by=parameters['by']
    
    n_a,n_x=Wax.shape
#     n_y=Wya.shape[0] # Though 'n_y' will be equal to 'n_x', I am mentioning both of them differently
    T_x=len(X)
    
    x=np.zeros((n_x,T_x))
    
    a=np.zeros((n_a,T_x+1))
    a[:,0]=a_prev[:,0]
    
    y_hat=np.zeros((n_x,T_x))
        
    loss=0
    
    for t in range(T_x):
        if (X[t] != None):
            x[X[t],t] = 1
        
        a_t,y_t=rnn_step_forward(x[:,t].reshape(-1,1),a[:,t].reshape(-1,1),parameters)
        
        a[:,t+1]=a_t[:,0]
        y_hat[:,t]=y_t[:,0]
        
        loss-=np.log(y_hat[Y[t],t])
        
    cache=(x,a,y_hat)
    return loss,cache

def rnn_step_backward(dz_t,x_t,a_t,a_prev,parameters,gradients): # Here Y is that timestep's correct index of character i.e Y[t]=Yt and                                                                         y_hat is that timestep's softmax output
#     y_hat_t=np.copy(y_hat)
#     dz_t=y_hat_t[Yt]-1
    gradients['dWya']+=np.dot(dz_t,a_t.T)
    gradients['dby']+=dz_t
    
    da=np.dot(parameters['Wya'].T,dz_t)+gradients['da_t']
    daraw=(1-a_t*a_t)*da
    
    gradients['dWaa']+=np.dot(daraw,a_prev.T)
    gradients['dWax']+=np.dot(daraw,x_t.T)
    gradients['dba']+=daraw
    
    gradients['da_t']=np.dot(parameters['Waa'],daraw)
    
    return gradients
    
def rnn_backward(X,Y,parameters,cache):
    Waa=parameters['Waa']
    Wax=parameters['Wax']
    Wya=parameters['Wya']
    ba=parameters['ba']
    by=parameters['by']
    
    (x,a,y_hat)=cache
    
    n_a,n_x=Wax.shape
    n_y=Wya.shape[0] # Though 'n_y' will be equal to 'n_x', I am mentioning both of them differently
    T_x=len(X)
    
    gradients={}
    gradients['dWya']=np.zeros_like(Wya)
    gradients['dby']=np.zeros_like(by)
    gradients['dWaa']=np.zeros_like(Waa)
    gradients['dWax']=np.zeros_like(Wax)
    gradients['dba']=np.zeros_like(ba)
    gradients['da_t']=np.zeros((n_a,1))
    
    for t in reversed(range(T_x)):
        dz_t=np.copy(y_hat[:,t])
        dz_t[Y[t]]-=1
        gradients=rnn_step_backward(dz_t.reshape(-1,1),x[:,t].reshape(-1,1),a[:,t+1].reshape(-1,1),a[:,t].reshape(-1,1),parameters,gradients)
        
    return gradients,a
