# -*- coding: utf-8 -*-
"""
Created on Sun Sep 24 23:16:42 2017

@author: Futami
"""

import gym
import numpy as np
import random
import matplotlib.pyplot as plt

env = gym.make('FrozenLake-v0')
#env = gym.make('CartPole-v0')


Q = np.zeros([env.observation_space.n,env.action_space.n])

l = .8#学習のステップ幅です
gamma = .95#割引率です。
num_episodes = 2000

rList = []
for i in range(num_episodes):
    s = env.reset()
    rAll = 0
    d = False
    j = 0
    while j < 99:#99歩歩いてもゴールにも落とし穴にも落ちない場合は終了します。
        #env.render()
        j+=1
        #Choose an action by greedily (with noise) picking from Q table
        a = np.argmax(Q[s,:] + np.random.randn(1,env.action_space.n)*(1./(i+1)))#greedyにアクションを選ぶことにします。但し探索ができるようにランダム性も入れておきます。このランダム性は徐々に減るようにしておきます。
        s1,r,d,_ = env.step(a)#アクションを入力し、次の状態や、報酬を生成します。
        Q[s,a] = Q[s,a] + l*(r + gamma*np.max(Q[s1,:]) - Q[s,a])#Q-learningの式に従ってテーブルを更新します。
        rAll += r#報酬の合計を増やします。
        s = s1#次の状態に移ります。
        if d == True:#もし落とし穴に落ちていたり、ゴールできていればそのエピソードを終了します。
            break
    rList.append(rAll)

print("Score over time: " +  str(sum(rList)/num_episodes))

print("action value table is :")
print(Q)