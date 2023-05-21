import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import random
import wandb
import argparse
import os 

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#loading the dataset
class MyDataset(Dataset):
    def __init__(self, csv_file):
        self.data = pd.read_csv(csv_file,names=["English","Hindi"],header=None)
        
    def __getitem__(self, index):
        x = self.data.iloc[index]["English"]
        y = self.data.iloc[index]["Hindi"]
        return x, y
    
    def __len__(self):
        return len(self.data)


train_data = MyDataset(os.getcwd()+'/hin_train.csv')
test_data = MyDataset(os.getcwd()+'/hin_test.csv')
val_data = MyDataset(os.getcwd()+'/hin_valid.csv')

ENGLEN=32
HINDILEN=32
BATCH_SIZE=128
englishwords=torch.full((len(train_data), ENGLEN), 2).to(device)
hindiwords=torch.full((len(train_data), HINDILEN), 2).to(device)

#creating the vocabulary
hindivocab=set()
englishvocab=set()
for x,y in train_data:
    for letter in x:
        englishvocab.add(letter)
    for letter in y:
        hindivocab.add(letter)  
for x,y in val_data:
    for letter in x:
        englishvocab.add(letter)
    for letter in y:
        hindivocab.add(letter)
for x,y in test_data:
    for letter in x:
        englishvocab.add(letter)
    for letter in y:
        hindivocab.add(letter)
hindivocab=list(hindivocab)
hindivocab.sort()
englishvocab=list(englishvocab)
englishvocab.sort()
#Adding start end pad characters to the vocab
hindivocab.insert(0,'0')#start
hindivocab.insert(1,'1') #end
hindivocab.insert(2,'2') #pad
englishvocab.insert(0,'0')#start
englishvocab.insert(1,'1') #end
englishvocab.insert(2,'2') #pad
#creating char to index and index to char dictionary
hindidictc={}
englishdictc={}
hindidicti={}
englishdicti={}
for i in range(len(hindivocab)):
    hindidicti[i]=hindivocab[i]
    hindidictc[hindivocab[i]]=i
for i in range(len(englishvocab)):
    englishdicti[i]=englishvocab[i]
    englishdictc[englishvocab[i]]=i

#converting train data into tensors
c=0
for x,y in train_data:
    for i in range(len(x)):
        englishwords[c][i]=englishdictc[x[i]]
    for i in range(len(y)):
        hindiwords[c][i]=hindidictc[y[i]]
    hindiwords[c][i+1]=1
    c+=1
#converting val data into tensors
englishwordsval=torch.full((len(val_data), ENGLEN), 2).to(device)
hindiwordsval=torch.full((len(val_data), HINDILEN), 2).to(device)
c=0
for x,y in val_data:
    for i in range(len(x)):
        englishwordsval[c][i]=englishdictc[x[i]]
    for i in range(len(y)):
        hindiwordsval[c][i]=hindidictc[y[i]]
    hindiwordsval[c][i+1]=1
    c+=1
#converting test data into tensors
englishwordstest=torch.full((len(test_data), ENGLEN), 2).to(device)
hindiwordstest=torch.full((len(test_data), HINDILEN), 2).to(device)
c=0
for x,y in test_data:
    for i in range(len(x)):
        englishwordstest[c][i]=englishdictc[x[i]]
    for i in range(len(y)):
        hindiwordstest[c][i]=hindidictc[y[i]]
    hindiwordstest[c][i+1]=1
    c+=1

#Encoder
class EncoderRNN(nn.Module):
    def __init__(self, input_size,hidden_size,embedding_size,num_layers,typecell,dropout,bidirectional):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.dropout = nn.Dropout(dropout)
        self.num_layers=num_layers
        self.embedding = nn.Embedding(input_size, embedding_size)
        self.bidirectional=bidirectional
        self.p=dropout
        self.typecell=typecell
        if self.typecell=="gru":
            self.step = nn.GRU(embedding_size, hidden_size,num_layers,dropout=self.p,bidirectional=self.bidirectional)    
        if self.typecell=="lstm":
            self.step = nn.LSTM(embedding_size, hidden_size,num_layers,dropout=self.p,bidirectional=self.bidirectional)
        if self.typecell=="rnn":
            self.step = nn.RNN(embedding_size, hidden_size,num_layers,dropout=self.p,bidirectional=self.bidirectional) 

    def forward(self, inp, hidden,cell=None):
        embedded = self.dropout(self.embedding(inp))
        #distinguishing between cell types
        #passing embedded eng  chars to get output and hidden
        if self.typecell=="gru":
            output, hidden = self.step(embedded, hidden)   
            return output,hidden
        if self.typecell=="rnn":
            output, hidden = self.step(embedded, hidden)   
            return output,hidden
        #lstm also returns cell, taken care or cell as well
        if self.typecell=="lstm": 
            output, (hidden,cell) = self.step(embedded, (hidden,cell))
            return output, (hidden,cell)

    def initHidden(self):
        #for bidirection
        num_layers=self.num_layers
        if self.bidirectional:
            #if bidirection is true then no of layers
            num_layers=self.num_layers*2
        hidden=torch.zeros(num_layers,BATCH_SIZE,self.hidden_size, device=device)
        if self.typecell=="lstm":
            #additional cell initialization for lstm
            cell=torch.zeros(num_layers,BATCH_SIZE,self.hidden_size, device=device)
            return (hidden,cell)        
        return hidden


    
class DecoderRNN(nn.Module):
    def __init__(self,input_size,hidden_size,embedding_size,num_layers,output_size,typecell,dropout):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.dropout = nn.Dropout(dropout)
        self.num_layers=num_layers
        self.embedding = nn.Embedding(input_size, embedding_size)
        self.output_size=output_size
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.Softmax(dim=2)
        self.p=dropout
        self.typecell=typecell
        if self.typecell=="gru":
            self.step = nn.GRU(embedding_size, hidden_size,num_layers,dropout=self.p)   
        if self.typecell=="rnn":
            self.step = nn.RNN(embedding_size, hidden_size,num_layers,dropout=self.p) 
        if self.typecell=="lstm":
            self.step = nn.LSTM(embedding_size, hidden_size,num_layers,dropout=self.p)

    def forward(self, inp, hidden,cell=None):
        embedded = self.dropout(self.embedding(inp))
        #passing embedded  predicted hindi chars to get output and hidden of a particular timestep
        if self.typecell=="gru":
            output, hidden = self.step(embedded, hidden)   
            output1=self.out(output)
            return output1,hidden
        if self.typecell=="rnn":
            output, hidden = self.step(embedded, hidden)   
            output1=self.out(output)
            return output1,hidden
        if self.typecell=="lstm": 
            output, (hidden,cell) = self.step(embedded,  (hidden,cell))
            output1=self.out(output)
            return output1, (hidden,cell)

    def initHidden(self):
        #for bidirection
        num_layers=self.num_layers
        hidden=torch.zeros(num_layers,BATCH_SIZE,self.hidden_size, device=device)
        if self.typecell=="lstm":
            #additional cell initialization for lstm
            cell=torch.zeros(num_layers,BATCH_SIZE,self.hidden_size, device=device)
            return (hidden,cell)        
        return hidden 
            
class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder,hencoder,cell=None):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.hencoder=hencoder
        self.cell=cell
    def forward(self, inp, target,teacher_force_ratio):
        #output tensor to store the outputs
        outputs = torch.zeros(HINDILEN,BATCH_SIZE ,len(hindivocab)).to(device)
        #encoder operation on eng letters
        if self.encoder.typecell=="lstm":
            p,(hencoder3d,cell)=self.encoder.forward(inp.to(device),self.hencoder,self.cell)
        else:
            p,hencoder3d=self.encoder.forward(inp.to(device),self.hencoder)
        #Operation to get hdecoder from outputs of encoder
        tempdecoder=torch.zeros(self.encoder.num_layers,BATCH_SIZE,hencoder3d.size()[2]).to(device)
        tempdecoder[0]=hencoder3d[hencoder3d.size()[0]//2-1]
        tempdecoder[1]=hencoder3d[(hencoder3d.size()[0]//2)*2-1]
        hdecoder=torch.add(tempdecoder[0],tempdecoder[1])
        hdecoder=hdecoder.repeat(self.decoder.num_layers,1,1)
        if self.encoder.typecell=="lstm":
            #same operation on cell as performed for hdecoder
            tempcell=torch.zeros(2,BATCH_SIZE,cell.size()[2]).to(device)
            tempcell[0]=hencoder3d[cell.size()[0]//2-1]
            tempcell[1]=hencoder3d[(cell.size()[0]//2)*2-1]
            cell=torch.add(tempcell[0],tempcell[1])
            cell=cell.repeat(self.decoder.num_layers,1,1)
        x=torch.full((1,BATCH_SIZE),hindidictc['0'])
        #passing start of word as input to decoder
        if self.encoder.typecell=="lstm":
            output,(hdecoder,cell)=self.decoder.forward(x.to(device),hdecoder,cell)
        else:
            output,hdecoder=self.decoder.forward(x.to(device),hdecoder)
        #storing the first output
        outputs[0]=output
        t=1
        #Looping through all the timesteps
        for i in range(1,HINDILEN):
            if random.random() > teacher_force_ratio:
                #passing the predicted output as input
                output=self.decoder.softmax(output)
                nextinp=torch.argmax(output, dim=2)
                if self.encoder.typecell=="lstm":
                    output,(hdecoder,cell)=self.decoder.forward(nextinp.to(device),hdecoder,cell)
                else:
                    output,hdecoder=self.decoder.forward(nextinp.to(device),hdecoder)
                outputs[t]=output
                t+=1
            else:
                #passing actual target of last timestep as input
                nextinp=target[i-1,:].unsqueeze(0)
                if self.encoder.typecell=="lstm":
                    output,(hdecoder,cell)=self.decoder.forward(nextinp.to(device),hdecoder,cell)
                else:
                    output,hdecoder=self.decoder.forward(nextinp.to(device),hdecoder)
                outputs[t]=output
                t+=1
        return outputs
        
    
def train(encoder,decoder,seq2seq,epoch):
    encoder_optimizer = optim.Adam(encoder.parameters(), lr=0.001)
    decoder_optimizer = optim.Adam(decoder.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss(reduction="sum")
    loss=0
    count=0
    numbatches=englishwords.shape[0]//BATCH_SIZE
    for ep in range(epoch):
        trainloss=0
        train_correct=0
        for i in range(numbatches):
            #zeroing the previous grad
            encoder_optimizer.zero_grad()
            decoder_optimizer.zero_grad()
            #getting the current batch
            temp=englishwords[i*BATCH_SIZE:(i+1)*BATCH_SIZE]
            temph=hindiwords[i*BATCH_SIZE:(i+1)*BATCH_SIZE]
            temp=temp.t()
            temph=temph.t()
            #getting output of the entire model
            output=seq2seq.forward(temp,temph,0.5)
            train_correct+=train_accuracy(output,temph)
            output = output[:].reshape(-1, output.shape[2])
            tem = temph[:].reshape(-1)
            #calculate the loss
            loss=criterion(output,tem)
            loss.backward()
            trainloss+=loss.item()
            #clip the grad
            torch.nn.utils.clip_grad_norm_(decoder.parameters(),max_norm = 1)
            torch.nn.utils.clip_grad_norm_(encoder.parameters(),max_norm = 1)
            encoder_optimizer.step()
            decoder_optimizer.step()
        #calculating validation accuracy
        val_correct,cur_loss=accuracy(seq2seq,englishwordsval,hindiwordsval)
        print(ep,trainloss/(51200*HINDILEN),cur_loss/(4096*HINDILEN),val_correct,train_correct)
        trainloss=trainloss/(51200*HINDILEN)
        cur_loss=cur_loss/(4096*HINDILEN)
        tra_acc=train_correct/51200
        val_acc=val_correct/4096
        wandb.log({"Training loss":trainloss,'Val loss':cur_loss,'Training Accuracy':tra_acc,'Val Accuracy':val_acc})

def train_accuracy(output,temph):
        output=nn.Softmax(dim=2)(output)
        output=torch.argmax(output,dim=2)
        temph=temph.t()
        output=output.t()
        correct=0
        for i in range(BATCH_SIZE):
            if(torch.equal(output[i],temph[i])):
                correct+=1
        return correct
        
def accuracy(seq2seq,english,hindi):
    numbatches=english.shape[0]//BATCH_SIZE
    correct=0
    criterion = nn.CrossEntropyLoss(reduction="sum")
    loss=0
    for i in range(numbatches):
        #getting the current batch
        temp=english[i*BATCH_SIZE:(i+1)*BATCH_SIZE]
        temph=hindi[i*BATCH_SIZE:(i+1)*BATCH_SIZE]
        temp=temp.t()
        temph=temph.t()
        #getting output of the entire model
        output=seq2seq.forward(temp,temph,0)
        o = output[:].reshape(-1, output.shape[2])
        tem = temph[:].reshape(-1)
        #calculate the loss
        x=criterion(o,tem)
        loss+=x.item()
        output=nn.Softmax(dim=2)(output)
        output=torch.argmax(output,dim=2)
        temph=temph.t()
        output=output.t()
        for i in range(BATCH_SIZE):
            if(torch.equal(output[i],temph[i])):
                correct+=1

    return correct,loss



parser=argparse.ArgumentParser()
parser.add_argument("-wp","--wandb_project",type=str,default="DLAssignment3")
parser.add_argument("-we","--wandb_entity",type=str,default="cs22m072")
parser.add_argument("-e","--epochs",type=int,default=15)
parser.add_argument("-b","--batch_size",type=int,default=128)
parser.add_argument("-ct","--cell_type",type=str,default="lstm")
parser.add_argument("-es","--embedding_size",type=int,default=512)
parser.add_argument("-hs","--hidden_size",type=int,default=512)
parser.add_argument("-nl","--num_layers",type=int,default=3)
parser.add_argument("-d","--dropout",type=float,default=0.2)
parser.add_argument("-bd","--bidirectional",type=str,default="Yes")
args=parser.parse_args()

wandb.login()
# wandb.init(project="DLAssignment3",entity="cs22m072")
wandb.init(args.wandb_project,args.wandb_entity)

wandb.run.name = "cellType_{}_embSize_{}_layers_{}_batchSize_{}_hidden_{}_dropout_{}".format(args.cell_type,args.embedding_size,args.num_layers,args.batch_size,args.hidden_size,args.dropout)
bidirectional=True
if(args.bidirectional=='Yes'):
        bidirectional=True
else:
        bidirectional=False
BATCH_SIZE=args.batch_size
encoder=EncoderRNN(len(englishvocab),args.hidden_size,args.embedding_size,args.num_layers,args.cell_type,args.dropout,bidirectional).to(device)
decoder=DecoderRNN(len(hindivocab),args.hidden_size,args.embedding_size,args.num_layers,len(hindivocab),args.cell_type,args.dropout).to(device)
if args.cell_type=='lstm':
        hencoder,cell=encoder.initHidden()
        seq2seq=Seq2Seq(encoder,decoder,hencoder,cell)
else:
        hencoder=encoder.initHidden()
        seq2seq=Seq2Seq(encoder,decoder,hencoder)
train(encoder,decoder,seq2seq,args.epochs)
