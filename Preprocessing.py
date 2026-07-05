import numpy as np

class label_encoder :  
    def __init__(self):
        self.encoded_labels={}

    def fit_transform(self,text_label):
        ulabels=list(sorted(set(text_label)))
        # print(ulabels)
        p=0
        n=len(ulabels)
        encode_val={}
        labels=[]
        while p<n:
            encode_val[ulabels[p]]=p
            p+=1
        print(encode_val)
        for i in text_label:
            labels.append(encode_val[i])
        self.encoded_labels=encode_val

        return np.array(labels)

class standard_scaler:
    def __init__(self):
        self.mean=None
        self.std=None

    def fit_transform(self,data):
        data = np.array(data)
        self.mean=np.mean(data,axis=0)
        self.std= np.std(data,axis=0)+1e-10
        return (data-self.mean)/self.std
    
    def transform(self,data):
        data = np.array(data)
        return (data-self.mean)/self.std

def _train_test_split(x,y,test_size=0.2,random_state=None):
        # x,y=np.array(x),np.array(y)
        if random_state!=None:
            np.random.seed(random_state)
        idx= np.arange(len(x))
        np.random.shuffle(idx)
        size=int(len(idx)*(1-test_size))
        _X,_x,_Y,_y= x.iloc[idx[:size]],x.iloc[idx[size:]],y.iloc[idx[:size]],y.iloc[idx[size:]]
        return _X,_x,_Y,_y
       