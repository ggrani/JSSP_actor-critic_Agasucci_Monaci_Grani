import torch
from torch import nn
from modules import CoreModel, LstmModel_policy, LstmModel_value

class Model():
    def __init__(self, D_in, specs, device = torch.device("cpu"), LSTMpolicyflag = False, LSTMvalueflag = False, TDflag = False, specs_ff = None, seed = None, edges = None, nnodes = None):
        self.device = device
        if edges is None:
            if LSTMpolicyflag == True:
                self.coremdl = LstmModel_policy(D_in, specs, self.device).to(self.device)
            elif LSTMvalueflag == True:
                self.coremdl = LstmModel_value(D_in, specs, specs_ff, self.device).to(self.device)
            else:
                self.coremdl = CoreModel(D_in, specs, self.device).to(self.device)
        if seed != None:
            torch.seed = seed

    def set_loss(self, losstype):
        if losstype == "mse":
            self.criterion = nn.MSELoss()
        elif losstype == "l1":
            self.criterion = nn.L1Loss()
        elif losstype == "smoothl1":
            self.criterion = nn.SmoothL1Loss()
        elif losstype == 'KLD':
            self.criterion = nn.KLDivLoss(reduction='batchmean')
        else:
            raise Exception("Invalid loss type")

    def set_optimizer(self, name, options):
        if name == "sgd":
            self.optimizer = torch.optim.SGD(self.coremdl.parameters(), lr=options["lr"], momentum=options["momentum"],nesterov=options["nesterov"])
        elif name == "adam":
            self.optimizer = torch.optim.Adam(self.coremdl.parameters(), lr=options["lr"])
        else:
            raise Exception("Invalid optimizer type")

    def set_scheduler(self, name, options):
        if name is None:
            self.scheduler = None
        elif name == "multiplicative":
            factor = options.get("factor") if options.get("factor") is not None else .99
            lmbda = lambda epoch : factor**epoch
            self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lmbda)

    def schedulerstep(self):
        if self.scheduler is not None:
            self.scheduler.step()

    def long_update_LSTM(self, x, y, nsteps, bsize = 0):
        for _ in range(nsteps):
            self.single_update_LSTM(x, y, bsize)

    def update_critic(self, x, rewards, nsteps):
        for _ in range(nsteps):
            values = self.coremdl(x)
            rewards = rewards.type(torch.FloatTensor).to(self.device)
            self.optimizer.zero_grad()
            self.criterion(values, rewards).backward()
            self.optimizer.step()

    def update_criticLSTM(self, x, rewards, nsteps, bsize = 0):
        for _ in range(nsteps):
            values = self.coremdl(x, bsize)
            rewards = rewards.type(torch.FloatTensor).to(self.device)
            self.optimizer.zero_grad()
            self.criterion(values, rewards).backward()
            self.optimizer.step()

    def apply_update(self, grad_flattened):
        n = 0
        for p in self.coremdl.parameters():
            numel = p.numel()
            g = grad_flattened[n:n + numel].view(p.shape)
            p.data += g
            n += numel

    def update_actor(self, policy_loss):
        self.optimizer.zero_grad()
        policy_loss.backward()
        self.optimizer.step()