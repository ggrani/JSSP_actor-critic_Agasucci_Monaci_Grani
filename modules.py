import torch
#torch.set_num_threads(8)
from torch import nn
from torch.nn.parameter import Parameter
from torch.nn import init
from torch.nn.utils.rnn import pad_sequence
import math
import numpy as np

class LogSumExp(nn.Module):
    def __init__(self,in_features, out_features,bias = False):
        super(LogSumExp, self).__init__()
        self.n = in_features
        self.N = out_features
        self.weight = Parameter(torch.Tensor(self.N, self.n))
        # if bias:
        #     self.bias = Parameter(torch.Tensor(out_features))
        # else:
        #     self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        # if self.bias is not None:
        #     fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
        #     bound = 1 / math.sqrt(fan_in)
        #     init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        size = x.size()
        if len(size) == 1:
            B = 1
            pW = torch.stack([torch.stack([ self.weight[i]*torch.as_tensor([x[ll] for ll in range(self.n)]) for b in range(B)])for i in range(self.N)])
            ret = torch.reshape(torch.logsumexp(pW, dim=2), (B,self.N))
        else:
            B = len(x)
            pW = torch.stack([torch.stack([self.weight[i] * torch.as_tensor([x[b][ll] for ll in range(self.n)]) for b in range(B)]) for i in range(self.N)])
            ret = torch.logsumexp(pW, dim=2).transpose(0,1)
        return ret


#superclass for the model that can be used inside the algorithm
class BasicModel(nn.Module):
    def __init__(self):
        super(BasicModel, self).__init__()
        self.layers = []
        self.depth = 0
        self.specialind = -1
    def forward(self, x):
        pass

    def _codify_layers(self,  i, specs, mB = None):
        spec0 = specs[i]
        spec = specs[i + 1]
        biasflag =  mB is None
        if spec[0] == "linear":
            self.layers.append(nn.Linear(spec0[1], spec[1], bias = biasflag))
        elif spec[0] == "relu":
            db = nn.Linear(spec0[1], spec[1], bias = biasflag)
            db2 = nn.ReLU()
            self.layers.append(db)
            self.layers.append(db2)
            self.depth = self.depth + 1
        elif spec[0] == "leakyrelu":
            db = nn.Linear(spec0[1], spec[1], bias = biasflag)
            db2 = nn.LeakyReLU()
            self.layers.append(db)
            self.layers.append(db2)
            self.depth = self.depth + 1
        elif spec[0] == "sigmoid":
            db = nn.Linear(spec0[1], spec[1], bias = biasflag)
            db2 = nn.Sigmoid()
            self.layers.append(db)
            self.layers.append(db2)
            self.depth = self.depth + 1
        elif spec[0] == "logsigmoid":
            db = nn.Linear(spec0[1], spec[1], bias = biasflag)
            db2 = nn.LogSigmoid()
            self.layers.append(db)
            self.layers.append(db2)
            self.depth = self.depth + 1
        elif spec[0] == "softmax":
            db = nn.Linear(spec0[1], spec[1], bias = biasflag)
            db2 = nn.Softmax(dim=0)
            self.layers.append(db)
            self.layers.append(db2)
            self.depth = self.depth + 1
        elif spec[0] == "logsumexp":
            db = LogSumExp(spec0[1], spec[1])
            self.layers.append(db)
        elif spec[0] == "approximator":
            val = 2 * round(spec[1] / 3)
            valres = spec[1] - val
            db = nn.Linear(spec0[1], valres)
            self.layers.append(db)
            self.specialLayer = nn.Linear(spec0[1], val)
            self.specialLayer2 = nn.Sigmoid()
            self.specialLayer3 = nn.Dropout()
            self.specialind = len(self.layers) - 1
        elif spec[0] == "maxpool":
            self.layers.append(nn.MaxPool1d(spec0[1], spec[1]))
        elif spec[0] == "softmin":
            self.layers.append(nn.Softmin(spec0[1], spec[1]))
        elif spec[0] == "self_relu":
            self.layers.append(selfPiecewise())
        else:
            raise Exception("Not a valid input class for the layer.")

class CoreModel(BasicModel):
    def __init__(self, D_in, specs, device):
        super(CoreModel, self).__init__()
        self.specialind = -1
        self.depth = len(specs)
        specs = [("", D_in)] + specs
        self.layers = []
        for i in range(self.depth):
            self._codify_layers(i, specs, mB = True)
        self.layers = nn.ModuleList(self.layers)
        print(self.layers)

        self.device = device

    def forward(self, x):
        y_pred = self.layers[0](x).to(self.device) #todo device?

        for i in range(1, self.depth):
            if i != self.specialind:
                # if i <= 2:
                #     with torch.no_grad():
                #         y_pred = self.layers[i](y_pred)
                y_pred = self.layers[i](y_pred)
            else:
                y_predA = self.layers[i](y_pred)
                y_predB = self.specialLayer(y_pred)
                y_predB = self.specialLayer2(y_predB)
                y_predB = self.specialLayer3(y_predB)
                y_pred = torch.cat((y_predA,y_predB), dim = 1)
        return y_pred

class LstmModel_policy(BasicModel): #o NNmodule
    def __init__(self, input_dim, specs_lstm_actor, device):
        super(LstmModel_policy, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = specs_lstm_actor[0]
        self.layer_dim = specs_lstm_actor[1]
        self.layer2_dim = specs_lstm_actor[2]
        self.device = device

        self.lstm = nn.LSTM(self.input_dim, self.hidden_dim, self.layer_dim, batch_first=True)
        self.lstm2 = nn.LSTM(self.hidden_dim, self.hidden_dim*2, self.layer2_dim, batch_first=True, bidirectional = False)

    def forward(self, xxm):
        xx = xxm[0] #instanza come matrice (diminuita)  [(1,56),(6,3),(-1,-1)][(-1,-1)][(1,23)(-1,-1)] (m,t)
        mask = torch.as_tensor(xxm[1], device = self.device) #maschera one hot   maschera [(0,0,1,56),(2,0,1,23)] -> [1,0,1] (job,pos,m,t)
        leng = max([len(xxi) for xxi in xx])
        lens = [leng - len(xxi) for xxi in xx]
        megapad = torch.as_tensor([xx[i] + [(0,0,0) for _ in range(lens[i])] for i in range(len(lens))], device = self.device) #padding tra i jobs, quindi lungo le macchine/operazioni
        minibatch = torch.zeros((1, megapad.size(0), self.hidden_dim), device = self.device) #numero jobs, 1, hidden_dim
        out = self.lstm(megapad)[0] #input (n jobs, n operations, n features), output: (n job, n operations, hidden dim)
        minibatch[0,:]=out[:,-1]  # (njobs, hidden_dim)
        out = self.lstm2(minibatch)[0] #input (1, n jobs, hidden dim), output: (1, n jobs, hidden dim2)
        out = torch.sum(out, axis=2) #(1, n jobs)
        y_pred = out.reshape(out.size(1)) * mask.float() #dubbio? [1,4,5,4]  [0,1,1,0] -> [0,4,5,0]
        exps = torch.max(torch.exp(y_pred - torch.max(y_pred)) * mask, 10 ** (-10) * mask) / (torch.max(10 ** (-10) * mask,(torch.exp(y_pred - torch.max(y_pred)) * mask)).sum())  # 0.0000000000000000001
        return exps

class LstmModel_value(BasicModel):
    def __init__(self, input_dim, specs_lstm_critic, specs_ff_critic, device):
        super(LstmModel_value, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = specs_lstm_critic[0]
        self.layer_dim = specs_lstm_critic[1]
        self.layer2_dim = specs_lstm_critic[2]
        self.device = device

        self.lstm = nn.LSTM(self.input_dim, self.hidden_dim, self.layer_dim, batch_first=True)
        self.lstm2 = nn.LSTM(self.hidden_dim, self.hidden_dim*2, self.layer2_dim, batch_first=True)#, bidirectional = True)
        self.network = CoreModel(self.hidden_dim*2, specs_ff_critic, self.device)

    def forward(self, xx, bsize = 0):
        if bsize == 0:
           xx = [xx]
        megapad = [torch.tensor(xxxi, device = self.device).type(torch.FloatTensor) for xxi in xx for xxxi in xxi]  # transofmo in tensore ogni job, diventa lista di tensori
        lens0 = len(xx[0])  # numero di jobs di ogni istanza
        megapad = pad_sequence(megapad, batch_first=True).to(self.device) #todo device  # padding tra i jobs, quindi lungo le macchine/operazioni
        megapad = [torch.tensor(np.asarray([megapad[i].cpu().numpy() for i in range(lens0*k, lens0*(k + 1))]), device = self.device) for k in range(len(xx))]
        megapad = torch.stack(megapad).to(self.device)
        minibatch = torch.zeros((megapad.size(0), megapad.size(1), self.hidden_dim), device = self.device)  # numero istanze, numero jobs di ogni istanza, hidden_dim
        for i in range(megapad.size(1)):
            inp = megapad[:,i]  # prendo il job i-esimo di ogni istanza, quindi sarà un input formato da (n_ist, n_macchine, n_features)
            out = self.lstm(inp)[0]  # , (h0.detach(), c0.detach()))  #(n_istanze, n_operations, hidden_dim) avrò per ogni job, n_macchine embeddings (?)
            minibatch[:, i] = out[:,-1]  # per ogni istanza, prendo l'embedding dell'ultima operazione di ogni job che mi può rappresentare una sorta di embedding del job, sintesi delle operazioni prima
        out = self.lstm2(minibatch)[0]  # prende in input minibatch e quindi restituisce (n_ist, n_jobs, hidden_dim*2) #??bah rifa un embedding di ogni job
        out = self.network(torch.sum(out, axis=1))  #(n_ist, hidden_dim*2) # sommo tra loro i jobs (non posso considerarne solo uno) come input
        return out  #(n_ist, 1)


class selfPiecewise(nn.Module):
    def __init__(self, a = 0.01):
        super(selfPiecewise, self).__init__()
        self._a = a
    def forward(self, x):
        first = x <= 0
        third = x > self._a
        second = torch.logical_not(first) * torch.logical_not(third)
        return third * (x/2.0 + (1-self._a/2.0)*torch.log10(torch.max(2.0*(x - self._a + 1.0), torch.ones_like(x)))) + second * (x * x) /(2*self._a)