# -*- coding: utf-8 -*-
"""
Created on Thu Nov  2 08:54:26 2017

@author: Futami
"""

from __future__ import print_function
import numpy as np
import tensorflow.contrib.slim as slim
import tensorflow as tf

import matplotlib.pyplot as plt
import seaborn

import gym
env = gym.make('CartPole-v0')


#これまでと同じようにエージェントのクラスを作成します。

#今回はエージェントは
#①ポリシーを学習する用のネットワーク
#②modelを学習する用のネットワークの２つから構成されています。

class agent():
    def __init__(self,H,mH,learning_rate):
        #####policy network
        self.observations = tf.placeholder(tf.float32, [None,4] , name="input_x")
        with tf.name_scope('policy_net'):
            layer1=slim.fully_connected(self.observations,H,weights_initializer=tf.contrib.layers.xavier_initializer(),biases_initializer=None,activation_fn=tf.nn.relu)
            self.probability=slim.fully_connected(layer1,1,weights_initializer=tf.contrib.layers.xavier_initializer(),biases_initializer=None,activation_fn=tf.nn.sigmoid)

        self.tvars = tf.trainable_variables()

        self.input_y = tf.placeholder(tf.float32,[None,1], name="input_y")
        self.advantages = tf.placeholder(tf.float32,name="reward_signal")

        adam = tf.train.AdamOptimizer(learning_rate=learning_rate)
        #今回も複数のエピソードをまとめて学習するので勾配をためておく容器を用意しておきます。
        self.W1Grad = tf.placeholder(tf.float32,name="batch_grad1")
        self.W2Grad = tf.placeholder(tf.float32,name="batch_grad2")
        batchGrad = [self.W1Grad,self.W2Grad]
        
        loglik = tf.log(self.input_y*(self.input_y - self.probability) + (1 - self.input_y)*(self.input_y + self.probability))
        loss = -tf.reduce_mean(loglik * self.advantages) 
        
        self.newGrads = tf.gradients(loss,self.tvars)
        #まとめた勾配を適用します。
        self.updateGrads = adam.apply_gradients(zip(batchGrad,self.tvars))

        
        ###model network
        #modelネットワークは5つの入力からなります。一つはアクション、四つは状態の特徴量です。
        self.previous_state = tf.placeholder(tf.float32, [None,5] , name="previous_state")
        with tf.name_scope('model_net'):
            layer1=slim.fully_connected(self.previous_state,mH,weights_initializer=tf.contrib.layers.xavier_initializer(),activation_fn=tf.nn.relu)            
            layer2=slim.fully_connected(layer1,mH,weights_initializer=tf.contrib.layers.xavier_initializer(),activation_fn=tf.nn.relu)
            #出力は状態を表す4つの特徴量と報酬、終了状態かどうかを表す確率の６つです。
            predicted_observation = slim.fully_connected(layer2,4,weights_initializer=tf.contrib.layers.xavier_initializer(),activation_fn=None)
            predicted_reward = slim.fully_connected(layer2,1,weights_initializer=tf.contrib.layers.xavier_initializer(),activation_fn=None)
            predicted_done = slim.fully_connected(layer2,1,weights_initializer=tf.contrib.layers.xavier_initializer(),activation_fn=tf.nn.sigmoid)
            

        
        
        self.true_observation = tf.placeholder(tf.float32,[None,4],name="true_observation")
        self.true_reward = tf.placeholder(tf.float32,[None,1],name="true_reward")
        self.true_done = tf.placeholder(tf.float32,[None,1],name="true_done")


        self.predicted_state = tf.concat([predicted_observation,predicted_reward,predicted_done],1)
        
        observation_loss = tf.square(self.true_observation - predicted_observation)
        reward_loss = tf.square(self.true_reward - predicted_reward)
        
        done_loss = tf.multiply(predicted_done, self.true_done) + tf.multiply(1-predicted_done, 1-self.true_done)
        done_loss = -tf.log(done_loss)

        self.model_loss = tf.reduce_mean(observation_loss + done_loss + reward_loss)
        
        modelAdam = tf.train.AdamOptimizer(learning_rate=learning_rate)
        self.updateModel = modelAdam.minimize(self.model_loss)



tf.reset_default_graph()
#ここからは実際の学習を回す箇所の実装を行います。
#学習のために、状態、報酬、アクション、終わったかどうかを表す（0,1）ことを記録する必要があります。そのリストを次のように４つ用意します。
xs,drs,ys,ds = [],[],[],[]

running_reward = None

gamma = 0.99 # 割引率
model_bs = 3 # policy networkを学習するときのBatch size 
real_bs = 3 # model networkを学習するときのBatch size 

H = 8 #policy network のhideen unitの数 
mH = 256 #model network のhideen unitの数 
learning_rate = 1e-2
myAgent = agent(H=H,mH=mH,learning_rate=learning_rate)

#いくつか学習の最中に良く使う機能を関数にまとめておきます。
#ためた勾配を0にリセットする関数です。
def resetGradBuffer(gradBuffer):
    for ix,grad in enumerate(gradBuffer):
        gradBuffer[ix] = grad * 0
    return gradBuffer

#割引報酬を計算する関数です。
def discount_rewards(r):
    discounted_r = np.zeros_like(r)
    running_add = 0
    for t in reversed(range(0, r.size)):
        running_add = running_add * gamma + r[t]
        discounted_r[t] = running_add
    return discounted_r


#この関数は、エージェントがひとつ手前のアクションと行動を与えられたときに、それを元に環境が返してくれる、次の（状態、報酬、終了状態かどうか）を計算する関数です。
def stepModel(sess, state, action):
    toFeed = np.reshape(np.hstack([state[-1][0],np.array(action)]),[1,5])
    myPredict = sess.run([myAgent.predicted_state],feed_dict={myAgent.previous_state: toFeed})
    reward = myPredict[0][:,4]
    observation = myPredict[0][:,0:4]
    #今回のモデルは位置は-2.4~2.4の間、角度も同様に範囲が決まっているので、clipをすることでその範囲からはみ出てしまった予測をむりやり、その範囲に落とし込むことにします。
    observation[:,0] = np.clip(observation[:,0],-2.4,2.4)
    observation[:,2] = np.clip(observation[:,2],-0.4,0.4)
    doneP = np.clip(myPredict[0][:,5],0,1)
    #donePは本来は確率ではなく、booleanのはずなのでここで変換します。
    if doneP > 0.1 or len(state)>= 300:
        done = True
    else:
        done = False
    return observation, reward, done


#初期条件などを入れておきます。
reward_sum = 0
episode_number = 1
real_episodes = 1

init = tf.global_variables_initializer()
batch_size = real_bs

#いくつか切り替え用のbooleanを用意します。
drawFromModel = False # これをＴｒｕｅにした場合、シミュレータではなく、学習したモデルから次の状況を再現するようにします。
trainTheModel = True # これをＴｒｕｅにした場合、modelの学習を行います。
trainThePolicy = False # これをTrueにした場合、policyの学習を行います。ここでFalseにしてると意味がないじゃないかと思われるかもしれませんが、modelの学習がある程度進んだらＴｒｕｅになるように後で条件文を加えて調整します。初めからmodel、policyを同時に学習させるのでは今回の場合は学習が安定しません。
switch_point = 1

# start!
with tf.Session() as sess:
    rendering = False
    sess.run(init)
    observation = env.reset()
    x = observation
    gradBuffer = sess.run(myAgent.tvars)
    #まだ学習が始まっていないので変数を取り出しても意味がないはずですが、勾配をためておく容器を用意するためにここで実行しています。
    gradBuffer = resetGradBuffer(gradBuffer)
    #今回も5000エピソード実行してみましょう
    while episode_number <= 5000:
        # 画面に表示する絵をどれぐらいの頻度で写すのかを決めておきます。
        if (reward_sum/batch_size > 150 and drawFromModel == False) or rendering == True : 
            env.render()
            rendering = True
            
        x = np.reshape(observation,[1,4])
        #状態を元にとる行動を決定します。
        tfprob = sess.run(myAgent.probability,feed_dict={myAgent.observations: x})
        action = 1 if np.random.uniform() < tfprob else 0

        # 状態と行動を記録しておきます。後で学習に必要になるためです。
        xs.append(x) 
        y = 1 if action == 0 else 0 
        ys.append(y)
        
        # 行動を受けて次の状態と、報酬、終わったかどうかを返すようにします。このときmodelから返すのか、シミュレータを使うのかを選択できるようにしておきます。
        if drawFromModel == False:
            observation, reward, done, info = env.step(action)
        else:
            observation, reward, done = stepModel(sess,xs,action)
                
        reward_sum += reward
        #おわったかどうかということと、rewardを記録しておきます。
        ds.append(done*1)
        drs.append(reward)
        
        #ひとつのエピソードが終わったなら学習を始めます。
        if done: 
            
            if drawFromModel == False: 
                real_episodes += 1
            episode_number += 1

            # リスト状態で保管していたのを学習で使えるように適切なnumpy arrayの形に変換します。
            epx = np.vstack(xs)
            epy = np.vstack(ys)
            epr = np.vstack(drs)
            epd = np.vstack(ds)
            xs,drs,ys,ds = [],[],[],[] # 容器は次のエピソードのためにリセットしましょう。
            
            #まずmodelの学習について考えます。
            if trainTheModel == True:
                actions = np.array([np.abs(y-1) for y in epy][:-1])
                state_prevs = epx[:-1,:]
                state_prevs = np.hstack([state_prevs,actions])
                state_nexts = epx[1:,:]
                rewards = np.array(epr[1:,:])
                dones = np.array(epd[1:,:])
                state_nextsAll = np.hstack([state_nexts,rewards,dones])

                feed_dict={myAgent.previous_state: state_prevs, myAgent.true_observation: state_nexts,myAgent.true_done:dones,myAgent.true_reward:rewards}
                loss,pState,_ = sess.run([myAgent.model_loss,myAgent.predicted_state,myAgent.updateModel],feed_dict)
                
            if trainThePolicy == True:
                discounted_epr = discount_rewards(epr).astype('float32')
                #学習の安定化のため報酬について正規化しておきます。
                discounted_epr -= np.mean(discounted_epr)
                discounted_epr /= np.std(discounted_epr)
                tGrad = sess.run(myAgent.newGrads,feed_dict={myAgent.observations: epx, myAgent.input_y: epy, myAgent.advantages: discounted_epr})
                # 勾配をためていきます。但し
                if np.sum(tGrad[0] == tGrad[0]) == 0:
                    break
                for ix,grad in enumerate(tGrad):
                    gradBuffer[ix] += grad
                              
            #勾配がある程度（今回は3エピソード分）たまったら更新します。
            if switch_point + batch_size == episode_number: 
                switch_point = episode_number
                if trainThePolicy == True:
                    sess.run(myAgent.updateGrads,feed_dict={myAgent.W1Grad: gradBuffer[0],myAgent.W2Grad:gradBuffer[1]})
                    gradBuffer = resetGradBuffer(gradBuffer)

                running_reward = reward_sum if running_reward is None else running_reward * 0.99 + reward_sum * 0.01
                if drawFromModel == False:
                    print('World Perf: Episode %f. Reward %f. action: %f. mean reward %f.' % (real_episodes,reward_sum/real_bs,action, running_reward/real_bs))
                    if reward_sum/batch_size > 200:
                        break
                reward_sum = 0

                # 100エピソードをこえたらポリシーをモデルから学習するということと、モデルをシミュレータから学習するということを交互に行います。
                if episode_number > 100:
                    drawFromModel = not drawFromModel
                    trainTheModel = not trainTheModel
                    trainThePolicy = not trainThePolicy
                    
            #次のエピソードの初期値を決めておきます。
            if drawFromModel == True:
                observation = np.random.uniform(-0.1,0.1,[4]) # モデルから学習するときは状態についてランダムに適当に決めておきます。
                batch_size = model_bs
            else:
                observation = env.reset()#シミュレータの場合はリセットをすればよいです。
                batch_size = real_bs
                
print(real_episodes)


#modelの学習状況について確認します。
plt.figure(figsize=(8, 12))
for i in range(6):
    #modelの予測をプロットします
    plt.subplot(6, 2, 2*i + 1)
    #グラフは上から順に4つのstate、報酬、booleanになっています。
    plt.plot(pState[:,i],c='b')
    #実際のシミュレータの動きをプロットします
    plt.subplot(6,2,2*i+1)
    plt.plot(state_nextsAll[:,i],c='g')
plt.tight_layout()
#filename = "row.pdf"
#plt.savefig(filename)