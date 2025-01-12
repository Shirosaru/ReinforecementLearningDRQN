
from gym_anytrading.datasets import   Bitstamp_BTCUSD_2017_2022_minute_Capital#L, FOREX_EURUSD_1H_ASK
import gym 
import gym_anytrading
import functools

import numpy as np
import chainer
import chainer.functions as F
import chainer.links as L
import chainerrl
from chainerrl.action_value import DiscreteActionValue
from chainerrl import experiments
from chainerrl import explorers
from chainerrl import misc
from chainerrl import replay_buffer
from tqdm import tqdm

#from chainer import Variable, optimizers, Chain, cuda

#custom_env = gym.make('forex-v0',
#               df = FOREX_EURUSD_1H_ASK,
#               window_size = 10,
#               frame_bound = (10, 300),
#               unit_side = 'right')

start_frame=10

custom_env = gym.make('stocks-v0',
                df = Bitstamp_BTCUSD_2017_2022_minute_Capital,
                window_size = start_frame,
                frame_bound = (start_frame, 3000000))


env=custom_env

print("env information:")
print("> shape:", env.shape)
print("> df.shape:", env.df.shape)
print("> prices.shape:", env.prices.shape)
print("> signal_features.shape:", env.signal_features.shape)
print("> max_possible_profit:", env.max_possible_profit())

print()
print("custom_env information:")
print("> shape:", custom_env.shape)
print("> df.shape:", custom_env.df.shape)
print("> prices.shape:", custom_env.prices.shape)
print("> signal_features.shape:", custom_env.signal_features.shape)
print("> max_possible_profit:", custom_env.max_possible_profit())


# 各パラメータの設定
df = Bitstamp_BTCUSD_2017_2022_minute_Capital
max_number_of_steps = 30000  #1試行のstep数(コマ数)
#max_number_of_steps = len(df)-1  #1試行のstep数(コマ数)
num_episodes = 300  #最大エピソード回数
gamma = 0.99


# Q-関数の定義
#class QFunction(chainer.Chain):
#    def __init__(self, obs_size, n_actions):
#        super(QFunction, self).__init__()
#        with self.init_scope():
#            n_hidden_channels = 100
#            obs_size=start_frame*2
#            n_hidden_channels_1 = 100
#            n_hidden_channels_2 = 50
#            self.l0=L.Linear(obs_size, n_hidden_channels_1)
#            self.l1=L.Linear(n_hidden_channels_1, n_hidden_channels_2)
#            self.l2=L.Linear(n_hidden_channels_2, n_hidden_channels_2)
#            self.l3=L.Linear(n_hidden_channels_2, n_actions)
#    def __call__(self, x, test=False):
#        h = F.tanh(self.l0(x))
#        h = F.tanh(self.l1(h))
#        h = self.l2(h)
#        return chainerrl.action_value.DiscreteActionValue(h)


#class QFunction(chainer.Chain):
#    def __init__(self, obs_size, n_actions):
#        super(QFunction, self).__init__()
#        n_hidden_channels = 100
#       with self.init_scope():
#def QFunction( obs_size, n_actions):
#        with chainerrl.links.StatelessRecurrentSequential():
#            obs_size=start_frame*2
#            n_hidden_channels_1 = 100            
#            n_hidden_channels_2 = 100            
#            n_hidden_channels_3 = 100            
#            n_hidden_channels_4 = 50   
#            self.L0=L.Convolution1D(obs_size , 64, ksize=1, stride=1, pad=1)
#            self.L1=L.Convolution1D(64, 64, ksize=1, stride=1, pad=1)
#            self.L2=L.Convolution1D(64, n_actions, ksize=1, stride=1, pad=1)
#            self.l0=L.Linear(obs_size, n_hidden_channels_1)
#            self.l1=L.Linear(n_hidden_channels_1, n_hidden_channels_2)
#            self.l2=L.Linear(n_hidden_channels_2, n_hidden_channels_3)
#            self.l3=L.Linear(n_hidden_channels_3, n_actions)
#            self.L3=functools.partial(F.reshape, shape=(-1, 3136))
#            self.L3=L.NStepLSTM(1, 3136, 512, 1)
    #        L.Linear(512, n_actions),L3=L.Linear(3136, 512),
#            self.L4=L.Linear(512, n_actions)
        
#    def __call__(self, x, test=False):
#            h = F.relu(self.L0(x))
#            h = F.relu(self.L1(h))
#            h = F.relu(self.L2(h))
#            h= functools.partial(F.reshape, shape=(-1, 3136))
#            h = F.relu(self.L3(h))
#            h = F.relu(self.L4(h))
#            h = F.relu(self.L5(h))
#            return chainerrl.action_value.DiscreteActionValue(self.L2(h))         
#        return chainerrl.links.StatelessRecurrentSequential(
#        L.Convolution1D(100, 32, 8, stride=4, pad=1),
#        F.relu,
#        L.Convolution1D(32, 64, 4, stride=2, pad=1),
#        F.relu,
#        L.Convolution1D(64, 64, 3, stride=1, pad=1),
#        functools.partial(F.reshape, shape=(-1, 3136)),
#        F.relu,
#        L.NStepLSTM(1, 3136, 512, 1),
#        L.Linear(512, n_actions),
#            L.NStepLSTM(1, 3136, 512, 0),
#            L.Linear(n_hidden_channels_4, n_actions),
#            DiscreteActionValue,
#            )
#    if args.recurrent:
        # Q-network with LSTM
n_actions = env.action_space.n
obs_size=start_frame*2
n_hidden_channels_1 = 1000            
n_hidden_channels_2 = 1000            
n_hidden_channels_3 = 500
#reshaped_channel = 400            
n_hidden_channels_4 = 200 

q_func = chainerrl.links.StatelessRecurrentSequential(

            L.Linear(obs_size, n_hidden_channels_1),
            F.relu,
            L.Linear(n_hidden_channels_1, n_hidden_channels_2),
            F.relu,
            L.Linear(n_hidden_channels_2, n_hidden_channels_3),

            F.relu,
            L.NStepLSTM(1, n_hidden_channels_3, n_hidden_channels_4, 0),
            F.relu,
            L.NStepLSTM(1, n_hidden_channels_4, n_hidden_channels_4, 0),
            L.Linear(n_hidden_channels_4, n_actions),
            DiscreteActionValue,
        )
replay_buffer = replay_buffer.EpisodicReplayBuffer(10 ** 6)




# 最適化アルゴリズムの設定

#q_func = QFunction(n_actions)
#q_func = QFunction(env.observation_space.shape[0], env.action_space.n)
optimizer = chainer.optimizers.Adam(eps=1e-3) # eps (input)：発散防止パラメータε
optimizer.setup(q_func)


# Uncomment to use CUDA
# q_func.to_gpu(0)

# ε-greedy法
explorer = chainerrl.explorers.LinearDecayEpsilonGreedy(
    start_epsilon=1.0, end_epsilon=0.1,
    decay_steps=num_episodes,
    random_action_func=env.action_space.sample) # ε-greedy法

# Experience Replay
#replay_buffer = chainerrl.replay_buffer.ReplayBuffer(capacity=10 ** 6, num_steps=30)

# DQN実行メソッド
#phi = lambda x: x.astype(np.float32, copy=False) # Chainerで利用できるようfloat32に変換
#agent = chainerrl.agents.DQN(
#    q_func, optimizer, replay_buffer, gamma, explorer,
#    replay_start_size=500, update_interval=1, target_update_interval=100, phi=phi)


#gpu_device = 0
#cuda.get_device(gpu_device).use()
#model.to_gpu(gpu_device)
#xp = cuda.cupy


#DoubleDqn
phi = lambda x: x.astype(np.float32, copy=False) # Chainerで利用できるようfloat32に変換
agent = chainerrl.agents.DoubleDQN(
    q_func, optimizer, replay_buffer, gamma, explorer, gpu=-1, 
    replay_start_size=500, update_interval=1, target_update_interval=100,
    minibatch_size=100, phi=phi, recurrent=True)


#observation=env.reset()

for episode in tqdm(range(num_episodes), total=num_episodes, leave=True):  #試行数分繰り返す
    observation = env.reset()
    done = False
    reward = 0
    R = 0

    for step in tqdm(range(max_number_of_steps), total=max_number_of_steps, leave=True):  #1試行のループ
########################################################
#        if episode % 30 == 0: # 30で割り切れたら
#            env.render()
######################################################
#        action = env.action_space.sample()
        action = agent.act_and_train(observation, reward) # トレーニング用の行動選択
        observation, reward, done, info = env.step(action)
        R += reward
        if done:
            break
    agent.stop_episode_and_train(observation, reward, done) # 結果の確認と新しいエピソードの準備

    # 学習済みモデルの保存
    agent.save('agent')

    if episode % 10 == 0:
        print('episode:', episode, 'Reward:', R, 'statistics:', agent.get_statistics(), 'Total Info', info, reward)



#while True:
#    action = env.action_space.sample()
#    observation, reward, done, info = env.step(action)
    # env.render()
#    if done:
#        print("info:", info)
#        break

#okay rendering slows down everything###########################

#import matplotlib.pyplot as plt
#plt.cla()
#env.render_all()
#plt.show()
######################################################
print("have i made it to the end of the file ?")
