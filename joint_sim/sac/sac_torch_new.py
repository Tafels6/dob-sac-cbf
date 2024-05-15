import os
import torch as T
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from sac.networks import ReplayBuffer
from sac.networks import ActorNetwork, CriticNetwork

class Agent():
    def __init__(self, lr=3e-4, input_dims=[8],
            env=None, gamma=0.99, n_actions=2, max_size=1000000, tau=5e-3, layer1_size=256, layer2_size=256, batch_size=256, reward_scale=1):
        self.gamma = gamma
        self.tau = tau
        self.lr = lr
        self.alpha = 1.  # TO DO: Adjust temperature
        self.memory = ReplayBuffer(max_size, input_dims, n_actions)
        self.batch_size = batch_size
        self.n_actions = n_actions
        self.max_action = env.action_space.high
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')

        # Target Entropy = −dim(A)
        self.target_entropy = -T.prod(T.Tensor(env.action_space.shape).to(self.device)).item()
        # changed for intersection
        # self.target_entropy = -T.prod(T.Tensor(env.action_space.shape).to(self.device)).item()**2
        self.log_alpha = T.zeros(1, requires_grad=True, device=self.device)
        self.alpha_optim = optim.Adam([self.log_alpha], lr=self.lr*0.5)

        self.actor = ActorNetwork(lr, input_dims, n_actions=n_actions, name='actor', max_action=env.action_space.high)
        self.critic_1 = CriticNetwork(lr, input_dims, n_actions=n_actions, name='critic_1')
        self.critic_2 = CriticNetwork(lr, input_dims, n_actions=n_actions, name='critic_2')
        self.critic_1_target = CriticNetwork(lr, input_dims, n_actions=n_actions, name='critic_target_1')
        self.critic_2_target = CriticNetwork(lr, input_dims, n_actions=n_actions, name='critic_target_2')
        self.scale = reward_scale
        self.update_network_parameters(tau=1) 

    def choose_action(self, observation):
        state = T.Tensor(np.array([observation])).to(self.actor.device)
        actions, _ = self.actor.sample_normal(state, reparameterize=False)
        return actions.cpu().detach().numpy()[0]

    def remember(self, state, action, reward, new_state, done):
        self.memory.store_transition(state, action, reward, new_state, done)

    def update_network_parameters(self, tau=None):  
        if tau is None:
            tau = self.tau

        critic_1_target_params = self.critic_1_target.named_parameters()
        critic_1_params = self.critic_1.named_parameters()
        critic_1_target_dict = dict(critic_1_target_params)
        critic_1_dict = dict(critic_1_params)
        
        for name in critic_1_dict: 
            critic_1_dict[name] = tau*critic_1_dict[name].clone() + (1-tau)*critic_1_target_dict[name].clone()
        self.critic_1_target.load_state_dict(critic_1_dict)
        
        critic_2_target_params = self.critic_2_target.named_parameters()
        critic_2_params = self.critic_2.named_parameters()
        critic_2_target_dict = dict(critic_2_target_params)
        critic_2_dict = dict(critic_2_params)

        for name in critic_2_dict:
            critic_2_dict[name] = tau*critic_2_dict[name].clone() + (1-tau)*critic_2_target_dict[name].clone()
        self.critic_2_target.load_state_dict(critic_2_dict)

    def learn(self, episode, step, global_step, writer, save_runs=False):
        if self.memory.mem_cntr < self.batch_size:
            return

        state, action, reward, state_, done = self.memory.sample_buffer(self.batch_size)

        state = T.tensor(state, dtype=T.float).to(self.actor.device)
        action = T.tensor(action, dtype=T.float).to(self.actor.device)
        reward = T.tensor(reward, dtype=T.float).to(self.actor.device)
        state_ = T.tensor(state_, dtype=T.float).to(self.actor.device)
        done = T.tensor(done).to(self.actor.device)
        
        with T.no_grad():
            action_, log_probs_ = self.actor.sample_normal(state_, reparameterize=True)
            qf1_next_target = self.critic_1_target.forward(state_, action_)
            qf1_next_target[done] = 0.0
            qf2_next_target = self.critic_2_target.forward(state_, action_)
            qf2_next_target[done] = 0.0
            
            min_qf_next_target = T.min(qf1_next_target, qf2_next_target) - self.alpha * log_probs_
            next_q_value = reward.unsqueeze(-1) + self.gamma * (min_qf_next_target)

        qf1 = self.critic_1.forward(state, action)
        qf2 = self.critic_2.forward(state, action)
        critic_1_loss = 0.5 * F.mse_loss(qf1, next_q_value)
        critic_2_loss = 0.5 * F.mse_loss(qf2, next_q_value)

        if save_runs:
            writer.add_scalar(f"Loss/Episode_{episode}/Step/critic_1_loss", critic_1_loss, step)
            writer.add_scalar(f"Loss/Episode_{episode}/Step/critic_2_loss", critic_2_loss, step)
            writer.add_scalar(f"Loss/Globle/critic_1_loss", critic_1_loss, global_step)
            writer.add_scalar(f"Loss/Globle/critic_2_loss", critic_2_loss, global_step)

        pi, log_pi = self.actor.sample_normal(state, reparameterize=True)
        log_pi = log_pi.unsqueeze(-1)
        qf1_pi = self.critic_1.forward(state, pi)
        qf2_pi = self.critic_2.forward(state, pi)
        min_qf_pi = T.min(qf1_pi, qf2_pi)
        actor_loss = (self.alpha * log_pi - min_qf_pi)
        actor_loss = T.mean(actor_loss)
        if save_runs:
            writer.add_scalar(f"Loss/Episode_{episode}/Step/actor_loss", actor_loss, step)
            writer.add_scalar(f"Loss/Globle/actor_loss", actor_loss, global_step)

        # alpha adaptive
        alpha_loss = -(self.log_alpha.exp() * (log_pi + self.target_entropy)).mean()
        if save_runs:
            writer.add_scalar(f"Loss/Episode_{episode}/Step/alpha_loss", alpha_loss, step)
            writer.add_scalar(f"Loss/Globle/alpha_loss", alpha_loss, global_step)
        self.alpha_optim.zero_grad()
        alpha_loss.backward(retain_graph=True)
        self.alpha_optim.step()
        self.alpha = self.log_alpha.exp()
        if save_runs:
            writer.add_scalar(f"alpha/Episode_{episode}/Step/alpha", self.alpha, step)
            writer.add_scalar(f"alpha/Globle/alpha", self.alpha, global_step)
        # print('alpha: ', self.alpha)

        self.actor.optimizer.zero_grad()
        actor_loss.backward(retain_graph=True)
        self.actor.optimizer.step()

        self.critic_1.optimizer.zero_grad()
        critic_1_loss.backward(retain_graph=True)
        self.critic_1.optimizer.step()
        
        self.critic_2.optimizer.zero_grad()
        critic_2_loss.backward(retain_graph=False)
        self.critic_2.optimizer.step()

        self.update_network_parameters()

    def save_models(self):
        print('.... saving models ....')
        self.actor.save_checkpoint()
        self.critic_1_target.save_checkpoint()
        self.critic_2_target.save_checkpoint()
        self.critic_1.save_checkpoint()
        self.critic_2.save_checkpoint()

    def load_models(self):
        print('.... loading models ....')
        self.actor.load_checkpoint()
        self.critic_1_target.load_checkpoint()
        self.critic_2_target.load_checkpoint()
        self.critic_1.load_checkpoint()
        self.critic_2.load_checkpoint()