import torch

class Agent:

    def __init__(self, latent_size, input_size, device="cpu"):
        """
        agent that make binary decisions
        :param latent_size: 8
        """
        self.latent_size = latent_size
        readout = torch.empty((latent_size, 1), device=device)
        # action is a linear readout of current latent states
        self.readout = torch.nn.Parameter(torch.nn.init.kaiming_normal(readout) * .01)
        self.rnn = torch.nn.GRUCell(input_size=input_size, hidden_size=latent_size, device=device)
        self.hidden = None
        self.state_history = []
        self.device = device
        self.train = True

    def step(self, input):
        self.hidden = self.rnn(input, self.hidden)
        readout = self.hidden @ self.readout # project hidden state to actions.
        self.state_history.append(self.hidden.detach().cpu().numpy())
        return readout

    def reset(self):
        self.hidden = None
        self.state_history = []

    def parameters(self):
        return list(self.rnn.parameters()) + [self.readout]


