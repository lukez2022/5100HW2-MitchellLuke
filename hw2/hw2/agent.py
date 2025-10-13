import os
import torch
import torch.optim as optim
from copy import deepcopy
from model import QNetwork, DuelingQNetwork
from gymnasium.wrappers import TimeLimit

class DQNAgent:
    def __init__(self, state_size, action_size, cfg, device='cuda'):
        self.device = device
        self.use_double = cfg.use_double
        self.use_dueling = cfg.use_dueling
        self.target_update_interval = cfg.target_update_interval
        q_model = DuelingQNetwork if self.use_dueling else QNetwork

        self.q_net = q_model(state_size, action_size, cfg.hidden_size, cfg.activation).to(self.device)
        self.target_net = deepcopy(self.q_net).to(self.device)
        self.optimizer = optim.AdamW(self.q_net.parameters(), lr=cfg.lr)

        self.tau = cfg.tau
        # update the gamma we use in the Bellman equation for n-step DQN
        self.gamma = cfg.gamma ** cfg.nstep

    def soft_update(self, target, source):
        """
        Soft update the target network using the source network
        """
        for target_param, source_param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_((1 - self.tau) * target_param.data + self.tau * source_param.data)

    @torch.no_grad()
    def get_action(self, state):
        """
        Get the action according to the current state and Q value
        """

        ############################
        # YOUR IMPLEMENTATION HERE #

        s = torch.as_tensor(state, dtype=torch.float, device=self.device).unsqueeze(0)
        q = self.q_net(s)
        action = int(torch.argmax(q, dim=1).item())

        ############################
        return action

    @torch.no_grad()
    def get_Q_target(self, state, action, reward, done, next_state) -> torch.Tensor:
        """
        Get the target Q value according to the Bellman equation
        """
        if self.use_double:
            # YOUR IMPLEMENTATION HERE
            next_q = self.q_net(next_state)
            a_star = torch.argmax(next_q, dim=1, keepdim=True)
            next_q_target = self.target_net(next_state).gather(1, a_star)
        else:
            # YOUR IMPLEMENTATION HERE
            next_q_target = self.target_net(next_state).max(dim=1, keepdim=True)[0]
        
        target = reward + (1 - done) * self.gamma * next_q_target
        return target

    def get_Q(self, state, action, use_double_net=False) -> torch.Tensor:
        """
        Get the Q value of the current state and action
        """
        ############################
        # YOUR IMPLEMENTATION HERE #
        
        if use_double_net == True:
            net = self.target_net
        else:
            net = self.q_net
        
        q_all = net(state)
        q_value = q_all.gather(1, action.long())
        return q_value
        ############################

    def update(self, batch, step, weights=None):
        state, action, reward, next_state, done = batch

        Q_target = self.get_Q_target(state, action, reward, done, next_state)

        Q = self.get_Q(state, action)

        if weights is None:
            weights = torch.ones_like(Q).to(self.device)

        td_error = torch.abs(Q - Q_target).detach()
        loss = torch.mean((Q - Q_target)**2 * weights)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if not step % self.target_update_interval:
            with torch.no_grad():
                self.soft_update(self.target_net, self.q_net)

        return loss.item(), td_error, Q.mean().item()

    def save(self, name):
        os.makedirs('models', exist_ok=True)
        torch.save(self.q_net.state_dict(), os.path.join('models', name))

    def load(self, name='best.pt'):
        self.q_net.load_state_dict(torch.load(os.path.join('models', name)))

    def __repr__(self) -> str:
        use_double = 'Double' if self.use_double else ''
        use_dueling = 'Dueling' if self.use_dueling else ''
        prefix = 'Normal' if not self.use_double and not self.use_dueling else ''
        return use_double + use_dueling + prefix + 'QNetwork'
