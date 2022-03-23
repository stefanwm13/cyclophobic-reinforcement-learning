import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable


def ortho_weights(shape, scale=1.):
    """ PyTorch port of ortho_init from baselines.a2c.utils """
    shape = tuple(shape)

    if len(shape) == 2:
        flat_shape = shape[1], shape[0]
    elif len(shape) == 4:
        flat_shape = (np.prod(shape[1:]), shape[0])
    else:
        raise NotImplementedError

    a = np.random.normal(0., 1., flat_shape)
    u, _, v = np.linalg.svd(a, full_matrices=False)
    q = u if u.shape == flat_shape else v
    q = q.transpose().copy().reshape(shape)

    if len(shape) == 2:
        return torch.from_numpy((scale * q).astype(np.float32))
    if len(shape) == 4:
        return torch.from_numpy((scale * q[:, :shape[1], :shape[2]]).astype(np.float32))


def atari_initializer(module):
    """ Parameter initializer for Atari models

    Initializes Linear, Conv2d, and LSTM weights.
    """
    classname = module.__class__.__name__

    if classname == 'Linear':
        module.weight.data = ortho_weights(module.weight.data.size(), scale=np.sqrt(2.))
        module.bias.data.zero_()

    elif classname == 'Conv2d':
        module.weight.data = ortho_weights(module.weight.data.size(), scale=np.sqrt(2.))
        module.bias.data.zero_()

    elif classname == 'LSTM':
        for name, param in module.named_parameters():
            if 'weight_ih' in name:
                param.data = ortho_weights(param.data.size(), scale=1.)
            if 'weight_hh' in name:
                param.data = ortho_weights(param.data.size(), scale=1.)
            if 'bias' in name:
                param.data.zero_()

class AtariCNN2(nn.Module):
    def __init__(self, num_actions):
        """ Basic convolutional actor-critic network for Atari 2600 games

        Equivalent to the network in the original DQN paper.

        Args:
            num_actions (int): the number of available discrete actions
        """
        super().__init__()

        self.conv = nn.Sequential(nn.Conv2d(3, 32, 8, stride=4),
                                  nn.ReLU(),
                                  nn.Conv2d(32, 64, 4, stride=2),
                                  nn.ReLU(),
                                  nn.Conv2d(64, 64, 3, stride=1),
                                  nn.ReLU())

        self.fc = nn.Sequential(nn.Linear(64 * 7 * 7, 2048),
                                nn.ReLU())

        self.pi = nn.Linear(2048, num_actions)
        self.v = nn.Linear(2048, 1)

        self.num_actions = num_actions

        # parameter initialization
        self.apply(atari_initializer)
        self.pi.weight.data = ortho_weights(self.pi.weight.size(), scale=.01)
        self.v.weight.data = ortho_weights(self.v.weight.size())

    def forward(self, conv_in):
        """ Module forward pass

        Args:
            conv_in (Variable): convolutional input, shaped [N x 4 x 84 x 84]

        Returns:
            pi (Variable): action probability logits, shaped [N x self.num_actions]
            v (Variable): value predictions, shaped [N x 1]
        """
        N = conv_in.size()[0]

        conv_out = self.conv(conv_in).view(N, 64 * 7 * 7)

        fc_out = self.fc(conv_out)

        pi_out = self.pi(fc_out)
        v_out = self.v(fc_out)

        return pi_out, v_out



class AtariCNN(nn.Module):
    def __init__(self, num_actions):
        """ Basic convolutional actor-critic network for Atari 2600 games

        Equivalent to the network in the original DQN paper.

        Args:
            num_actions (int): the number of available discrete actions
        """
        super().__init__()

        self.conv = nn.Sequential(nn.Conv2d(3, 32, 8, stride=4),
                                  nn.ReLU(),
                                  nn.Conv2d(32, 64, 4, stride=2),
                                  nn.ReLU(),
                                  nn.Conv2d(64, 64, 3, stride=1),
                                  nn.ReLU())

        self.fc = nn.Sequential(nn.Linear(64 * 7 * 7, 512),
                                nn.ReLU())

        self.pi = nn.Linear(512, num_actions)
        self.v_ex = nn.Linear(512, 1)
        
        self.num_actions = num_actions

        # parameter initialization
        self.apply(atari_initializer)
        self.pi.weight.data = ortho_weights(self.pi.weight.size(), scale=.01)
        self.v_ex.weight.data = ortho_weights(self.v_ex.weight.size())


    def forward(self, conv_in):
        """ Module forward pass

        Args:
            conv_in (Variable): convolutional input, shaped [N x 4 x 84 x 84]

        Returns:
            pi (Variable): action probability logits, shaped [N x self.num_actions]
            v (Variable): value predictions, shaped [N x 1]
        """
        N = conv_in.size()[0]

        conv_out = self.conv(conv_in).reshape(N, 64 * 7 * 7)
        #print(conv_out.shape)


        fc_out = self.fc(conv_out)
        #print(fc_out.shape)


        pi_out = self.pi(fc_out)
        #print(pi_out.shape)
        
        v_ex_out = self.v_ex(fc_out)

        return pi_out, v_ex_out
    
    
    
class AtariCNNLSTM(nn.Module):
    def __init__(self, num_actions):
        """ Basic convolutional actor-critic network for Atari 2600 games

        Equivalent to the network in the original DQN paper.

        Args:
            num_actions (int): the number of available discrete actions
        """
        super().__init__()

        self.conv = nn.Sequential(nn.Conv2d(3, 32, 8, stride=4),
                                  nn.ReLU(),
                                  nn.Conv2d(32, 64, 4, stride=2),
                                  nn.ReLU(),
                                  nn.Conv2d(64, 64, 3, stride=1),
                                  nn.ReLU())

        self.fc = nn.Sequential(nn.Linear(64 * 7 * 7, 512),
                                nn.ReLU())

        self.lstm = nn.LSTM(512, 256, 1, batch_first=True)
        self.hidden_state = torch.randn(1, 1, 256).cuda()
        self.cell_state = torch.randn(1,1,256).cuda()
        self.hidden = (self.hidden_state, self.cell_state)

        self.pi = nn.Linear(256, num_actions)
        self.v = nn.Linear(256, 1)

        self.num_actions = num_actions

        # parameter initialization
        self.apply(atari_initializer)
        self.pi.weight.data = ortho_weights(self.pi.weight.size(), scale=.01)
        self.v.weight.data = ortho_weights(self.v.weight.size())

    def forward(self, conv_in, hidden):
        """ Module forward pass

        Args:
            conv_in (Variable): convolutional input, shaped [N x 4 x 84 x 84]

        Returns:
            pi (Variable): action probability logits, shaped [N x self.num_actions]
            v (Variable): value predictions, shaped [N x 1]
        """
        N = conv_in.size()[0]

        conv_out = self.conv(conv_in).reshape(N, 64 * 7 * 7)

        fc_out = self.fc(conv_out)
        
        inp = fc_out.unsqueeze(1)

        out, hidden = self.lstm(inp, hidden)
        
        pi_out = self.pi(out.squeeze(1))
        v_out = self.v(out.squeeze(1))

        return pi_out, v_out, hidden
    
    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        hidden = (weight.new(1, 1, 256).zero_().to(torch.device("cuda")),
                      weight.new(1, 1, 256).zero_().to(torch.device("cuda")))
        return hidden
