# -*- coding: utf-8 -*-
"""
Created on Thu Sep 28 10:48:36 2017

@author: Futami
"""

from __future__ import division

import gym
import numpy as np
import random
import tensorflow as tf
import matplotlib.pyplot as plt

env = gym.make('FrozenLake-v0')

tf.reset_default_graph()

#Q関数の関数近似とアクションの定義
#入力として16種類のState
'''
inputs1 = tf.placeholder(shape=[1,env.observation_space.n],dtype=tf.float32)
W = tf.Variable(tf.random_uniform([env.observation_space.n,4],0,0.01))
b = tf.Variable(tf.random_uniform([4],0,0.01))
W2 = tf.Variable(tf.random_uniform([4,4],0,0.01))
b2 = tf.Variable(tf.random_uniform([4],0,0.01))

#出力として4種類のアクション
H = tf.matmul(inputs1,W)+b
H=tf.nn.relu(H)
Qout = tf.matmul(H,W2)+b2#1*4が出る
'''
inputs1 = tf.placeholder(shape=[1,env.observation_space.n],dtype=tf.float32)
W = tf.Variable(tf.random_uniform([env.observation_space.n,4],0,0.01))
Qout = tf.matmul(inputs1,W)          
#その中で最大のものを予想とされるアクションとして出力
predict = tf.argmax(Qout,1)

#Q関数の勾配降下による学習の定義
nextQ = tf.placeholder(shape=[1,4],dtype=tf.float32)
loss = tf.reduce_sum(tf.square(nextQ - Qout))
trainer = tf.train.GradientDescentOptimizer(learning_rate=0.1)
updateModel = trainer.minimize(loss)

init = tf.global_variables_initializer()

gamma = .99
epsilon = 0.1
num_episodes = 2000

jList = []
rList = []
with tf.Session() as sess:
    sess.run(init)
    for i in range(num_episodes):

        s = env.reset()
        rAll = 0
        d = False
        j = 0

        while j < 99:
            j+=1
            #greedyなアクションの選択
            a,allQ = sess.run([predict,Qout],feed_dict={inputs1:np.identity(16)[s:s+1]})
            #epsilon-greedyの確率を計算
            if np.random.rand(1) < epsilon:
                a[0] = env.action_space.sample()
            #アクションを環境に対して行い、次の状態s1、報酬rなどを手に入れます。
            s1,r,d,_ = env.step(a[0])
            #次の状態s1における行動価値関数を計算
            Q1 = sess.run(Qout,feed_dict={inputs1:np.identity(16)[s1:s1+1]})

            #Qlearningの式の部分を書きます
            maxQ1 = np.max(Q1)
            targetQ = allQ
            targetQ[0,a[0]] = r + gamma*maxQ1
            #次に勾配降下で1ステップ学習します
            _,W1 = sess.run([updateModel,W],feed_dict={inputs1:np.identity(16)[s:s+1],nextQ:targetQ})
            rAll += r
            s = s1
            if d == True:
                #ε-greedyのεの値を下げます。
                epsilon = 1./((i/50) + 10)
                break
        jList.append(j)
        rList.append(rAll)
print("Percent of succesful episodes: " + str(sum(rList)/num_episodes) + "%")

#plt.plot(rList)
#plt.plot(jList)