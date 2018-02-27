# -*- coding: utf-8 -*-
"""
Created on Wed Nov  1 17:53:37 2017

@author: Futami
"""

import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
import gym
import matplotlib.pyplot as plt

env = gym.make('CartPole-v0')


gamma = 0.99

def discount_rewards(r):
    #割引率をかける操作
    discounted_r = np.zeros_like(r)
    running_add = 0
    #reversed
    for t in reversed(range(0, r.size)):
        #最新のはrunning_add=r[final](=r_f)
        #次のはgamma*r_f+r_(f-1)
        #その次はgamma^2*r_f+gamma*r_(f-1)+r_(f-2)
        running_add = running_add * gamma + r[t]
        discounted_r[t] = running_add
    #結局 G=r_start+gamma*r_(s+1)+gamma^2*r_(s+2)+....が返される
    return discounted_r

class agent():
    def __init__(self, lr, s_size,a_size,h_size):
        #①を実装します。つまり状態を入力とし、input-hidden-hidden-softmax-outputというニューラルネットワークを使って各アクションをとる確率を出力します。
        self.state_in= tf.placeholder(shape=[None,s_size],dtype=tf.float32)
        hidden = slim.fully_connected(self.state_in,h_size,biases_initializer=None,activation_fn=tf.nn.relu)
        self.output = slim.fully_connected(hidden,a_size,activation_fn=tf.nn.softmax,biases_initializer=None)
        #最大の確率となる番号を出力します。
        self.chosen_action = tf.argmax(self.output,1)

        #②を実装します。つまり勾配の計算を実装します。このためには実際にとった行動とリワードを入力する必要があります。
        self.reward_holder = tf.placeholder(shape=[None],dtype=tf.float32)
        self.action_holder = tf.placeholder(shape=[None],dtype=tf.int32)

        self.indexes = tf.range(0, tf.shape(self.output)[0]) * tf.shape(self.output)[1] + self.action_holder
        self.responsible_outputs = tf.gather(tf.reshape(self.output, [-1]), self.indexes)

        self.loss = -tf.reduce_mean(tf.log(self.responsible_outputs)*self.reward_holder)
        
        tvars = tf.trainable_variables()
        self.gradient_holders = []
        for idx,var in enumerate(tvars):
            placeholder = tf.placeholder(tf.float32,name=str(idx)+'_holder')
            self.gradient_holders.append(placeholder)

        self.gradients = tf.gradients(self.loss,tvars)
        #③を実装します。つまり実際に更新を行う箇所です。tensorflowではapply_gradientsで簡単に記述できます。ステップ幅は今回はAdamに任せることにします。
        optimizer = tf.train.AdamOptimizer(learning_rate=lr)
        self.update_batch = optimizer.apply_gradients(zip(self.gradient_holders,tvars))



tf.reset_default_graph()
#定義したエージェントに定数を代入します。
myAgent = agent(lr=1e-2,s_size=env.observation_space.shape[0],a_size=env.action_space.n,h_size=8)
total_episodes = 5000 #今回はとりあえずこの程度のエピソードを実行します。
max_ep = 999 #ひとつのエピソードの長さの限度を設定します。

#毎エピソードごとに更新するということはしません。5回のエピソードが終わるごとに更新します。
update_frequency = 5

init = tf.global_variables_initializer()


with tf.Session() as sess:
    sess.run(init)
    i = 0
    total_reward = []
    total_lenght = []
    
    #複数文のエピソードをまとめて更新するので、その分の更新の勾配をためておく器を用意しておきます。
    grad_holder = sess.run(tf.trainable_variables())
    for ix,grad in enumerate(grad_holder):
        grad_holder[ix] = grad * 0

    while i < total_episodes:
        env.render()
        s = env.reset()
        running_reward = 0
        ep_history = []
        #max_epに達するか棒がこけるかするまでエピソードを回します。
        for j in range(max_ep):
            #状態を入力として、各行動をとる確率を出します。
            a_dist = sess.run(myAgent.output,feed_dict={myAgent.state_in:[s]})
            #どの行動をとるかその確率を元にランダムに選択します。
            a = np.random.choice(a_dist[0],p=a_dist[0])
            a = np.argmax(a_dist == a)
            #選んだ行動をして、環境と収益を受け取ります。
            s1,r,d,_ = env.step(a)
            #どんな行動をしたのかなどを一応記録します。
            ep_history.append([s,a,r,s1])
            s = s1
            running_reward += r
            #棒がこけたらパラメータの調整をします。
            if d == True:
                ep_history = np.array(ep_history)
                #報酬を割り引きます。
                ep_history[:,2] = discount_rewards(ep_history[:,2])
                feed_dict={myAgent.reward_holder:ep_history[:,2],
                        myAgent.action_holder:ep_history[:,1],myAgent.state_in:np.vstack(ep_history[:,0])}
                grads = sess.run(myAgent.gradients, feed_dict=feed_dict)
                for idx,grad in enumerate(grads):
                    grad_holder[idx] += grad
                
                #更新するのに十分なエピソードが立っていれば更新を行います。
                if i % update_frequency == 0 and i != 0:
                    feed_dict= dictionary = dict(zip(myAgent.gradient_holders, grad_holder))
                    _ = sess.run(myAgent.update_batch, feed_dict=feed_dict)
                    for ix,grad in enumerate(grad_holder):
                        grad_holder[ix] = grad * 0
                #報酬の合計値および何ステップ立っていられたかの情報を記入します。         
                total_reward.append(running_reward)
                total_lenght.append(j)
                break


            #直近の100秒でどれぐらいの報酬をためられたのかをチェックします。
        if i % 100 == 0:
            print(np.mean(total_reward[-100:]))
        i += 1