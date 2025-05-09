import torch.nn as nn
import torch.nn.functional as F
from utils import weights_init_normal
import torch

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        # A bunch of convolutions one after another                    
        model = [   nn.Conv2d(3, 32, 4, stride=2, padding=1),
                    nn.BatchNorm2d(32),
                    nn.LeakyReLU(0.2, inplace=True),
                    nn.Conv2d(32, 64, 4, stride=2, padding=1)]
                    
        model += [  nn.BatchNorm2d(64),
                    nn.LeakyReLU(0.2, inplace=True),
                    nn.Conv2d(64, 128, 4, stride=2, padding=1)]
                    
        model += [  nn.BatchNorm2d(128),
                    nn.LeakyReLU(0.2, inplace=True),
                    nn.Conv2d(128, 256, 4, stride=2, padding=1)]
                    
        model += [  nn.BatchNorm2d(256),
                    nn.LeakyReLU(0.2, inplace=True),
                    nn.Conv2d(256, 256, 4, stride=2, padding=1)]
                    
        model += [  nn.BatchNorm2d(256),
                    nn.LeakyReLU(0.2, inplace=True),
                    nn.Conv2d(256, 256, 4, stride=2, padding=1)]
                    

        self.relu = nn.LeakyReLU(0.2, inplace=True)
        self.model = nn.Sequential(*model)
        self.fc1 = torch.nn.Linear(16384, 1024)
        self.fc2 = torch.nn.Linear(1024, 1024)
        self.fc3 = torch.nn.Linear(1024, 2)

    def forward(self,x):
        x=self.model(x)
        x=x.view(x.size(0),-1)
        x=self.relu(x)
        x=self.fc1(x)
        x=self.relu(x)
        x=self.fc2(x)
        x=self.relu(x)
        x=self.fc3(x)
        return x
        '''
        x=self.model(x)
        x=x.view(x.size(0),-1)
        x=self.fc2(x)
        x=self.fc3(x)
        x=self.fc4(x)
        k1=x[:,0]
        b1=x[:,1]
        k2=x[:,2]
        b2=x[:,3]
        xx=(b2-b1/(k1-k2+1e-8)).unsqueeze(1)
        yy=(k1*(b2-b1)/(k1-k2+1e-8)+b1).unsqueeze(1)
        
        return torch.cat((yy,xx),1)
        '''
