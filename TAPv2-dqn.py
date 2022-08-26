import tensorflow as tf
import numpy as np
import random
from collections import deque


# Hyper Parameters for DQN
GAMMA = 0.9 # discount factor for target Q
INITIAL_EPSILON = 0.5 # starting value of epsilon
FINAL_EPSILON = 0.01 # final value of epsilon
REPLAY_SIZE = 10000 # experience replay buffer size
BATCH_SIZE = 32 # size of minibatch

class DQN():
  # DQN Agent
  def __init__(self, env):
    # init experience replay
    self.replay_buffer = deque()
    # init some parameters
    self.time_step = 0
    self.Q_value_save = [0,0,0,0,0,0]
    self.epsilon = INITIAL_EPSILON
    self.state_dim = env.observation_space.shape[0] #shape[0]:一维时是列数（元素个数），二维时是行数
    self.action_dim = env.action_space.n

    self.create_Q_network()
    self.create_training_method()
    
    self.saver = tf.train.Saver() 
    # Init session
    self.session = tf.InteractiveSession()
    self.session.run(tf.global_variables_initializer())
    self.saver = tf.train.Saver()
    
    Q_value = self.Q_value.eval(feed_dict = {
      self.state_input:[np.array([300,140,50,40])]
      })[0]
    print('the very begining Q_value:',Q_value) # 应该是随机的

  def create_Q_network(self):
    # 网络权重
    W1 = self.weight_variable([self.state_dim,20])
    b1 = self.bias_variable([20])
    W2 = self.weight_variable([20,self.action_dim])
    b2 = self.bias_variable([self.action_dim])
    # 输入层
    self.state_input = tf.placeholder("float",[None,self.state_dim]) # 行数不固定，列数为状态空间维度
    # 隐藏层
    h_layer = tf.nn.relu(tf.matmul(self.state_input,W1) + b1) # relu()将矩阵中小于0的设置为0（relu函数）； matmul矩阵乘；
    # Q Value层
    self.Q_value = tf.matmul(h_layer,W2) + b2 # (隐藏层 * 权重W2 ) + bias_action

  def create_training_method(self):
    self.action_input = tf.placeholder("float",[None,self.action_dim]) # one hot presentation 独热
    self.y_input = tf.placeholder("float",[None])
    Q_action = tf.reduce_sum(tf.multiply(self.Q_value,self.action_input),reduction_indices = 1) # reduce_sum 用于计算tensor沿着某一维度的和，可以在求和后降维;如果不指定维则计算所有元素总和; multiply 元素直接相乘；
    self.cost = tf.reduce_mean(tf.square(self.y_input - Q_action)) # 损失函数 L(w) = E[(r+ymax_a'Q(s',a',w)-Q(s,a,w))^2]
    self.optimizer = tf.train.AdamOptimizer(0.0001).minimize(self.cost)  # Adam算法的优化器,learning_rate=0.0001; .minimize: 通过更新 var_list 添加操作以最大限度地最小化 loss

  def perceive(self,state,action,reward,next_state,done):
    one_hot_action = np.zeros(self.action_dim)
    one_hot_action[action] = 1
    self.replay_buffer.append((state,one_hot_action,reward,next_state,done)) # 加入经验回放池
    if len(self.replay_buffer) > REPLAY_SIZE:
      self.replay_buffer.popleft() # 回放池满了pop掉最早的

    if len(self.replay_buffer) > BATCH_SIZE:
      self.train_Q_network() # 回放池达到batch_size就去训练

  def train_Q_network(self):
    self.time_step += 1
    # Step 1: obtain random minibatch from replay memory
    minibatch = random.sample(self.replay_buffer,BATCH_SIZE) #sample：从buffer里面随机采样BATCH_SIZE个样本作为minibatch序列（不改变原序列）
    state_batch = [data[0] for data in minibatch] # vscode 迷之报错，这分明没错…
    action_batch = [data[1] for data in minibatch]
    reward_batch = [data[2] for data in minibatch]
    next_state_batch = [data[3] for data in minibatch]

    # Step 2: calculate y
    y_batch = []
    Q_value_batch = self.Q_value.eval(feed_dict={self.state_input:next_state_batch})  # Python中eval() 函数用来执行一个字符串表达式，并返回表达式的值; TensorFlow中eval()其实就是tf.Tensor的Session.run()的另一种写法，eval()只能用于tf.tensor类对象，也就是有输出的operaton。没有输出的operation，使用session.run()
    # placeholder定义输入值的类型，以及数据结构, feed_dict用于接收真实的输入值（python中dist是字典，元素是键值对的存储模式）
    # 即state_input = next_state_batch

    # 从经验回放集合D中采样m个样本{ϕ(Sj),Aj,Rj,ϕ(S′j),is_endj},j=1,2.,,,m，计算当前目标Q值yj：
    for i in range(0,BATCH_SIZE):
      done = minibatch[i][4] # 即is_endj
      if done:
        y_batch.append(reward_batch[i]) # 最后一步，没有下一步，不需要加GAMMA * np.max(Q(s',a',w))
      else :
        y_batch.append(reward_batch[i] + GAMMA * np.max(Q_value_batch[i]))

    self.optimizer.run(feed_dict={
      self.y_input:y_batch,
      self.action_input:action_batch,
      self.state_input:state_batch
      })

  #在Q网络中使用 当前状态序列中状态S的特征向量ϕ(S) 作为输入，得到Q网络的所有动作对应的Q值输出。用ϵ−贪婪法在当前Q值输出中选择对应的动作A
  def egreedy_action(self,state):
    # 获得当前状态下Q网络所有动作对应的Q值输出
    Q_value = self.Q_value.eval(feed_dict = {
      self.state_input:[state]
      })[0]
    self.Q_value_save = Q_value
    # ϵ−贪婪
    if random.random() <= self.epsilon:
        # 小于ϵ，选择随机action，并逐步减小ϵ
        self.epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / 10000
        return random.randint(0,self.action_dim - 1) #randint(a, b) 随机生成[a,b]区间的整数；随机生成[0,action维度-1]的随机整数用于选择随机action
    else:
        # 大于ϵ，选择当前最优策略，即最大Q值对应的action，并逐步减小ϵ
        self.epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / 10000
        return np.argmax(Q_value) #np.argmax 取出最大值对应的索引

  def action(self,state):
    return np.argmax(self.Q_value.eval(feed_dict = {
      self.state_input:[state]
      })[0])

  def weight_variable(self,shape):
    #initial = tf.truncated_normal(shape) # 产生截断正态分布随机数 truncated_normal(shape,mean=0.0,stddev=1.0,dtype=tf.float32,seed=None,name=None) 取值范围为 [ mean - 2 * stddev, mean + 2 * stddev ]
    initial = tf.truncated_normal(shape, mean = 0.0, stddev = 1)
    return tf.Variable(initial)

  def bias_variable(self,shape):
    initial = tf.constant(0.01, shape = shape)
    return tf.Variable(initial)

class space():
  def __init__(self,space_type):
    if space_type == 0:
      self.the_action_space = np.array((0,1,2,3,4,5)) #th0 -20,0,+20 th1 -20.0,+20
      self.n = len(self.the_action_space)
    #elif space_type == 1:
    #  self.the_observation_space = np.array(np.zeros((4,1)))

class TAP_env():
  # environemnt of TAP
  # 
  def __init__(self, GEN_DELAY_AMOUNT,GEN_DELAY_MIN,GEN_DELAY_AVG):
    print("Initializing env...")
    self.draw_pic_reward_total = deque()
    self.draw_pic_acc_total = deque()
    self.draw_pic_acc_total_test = deque()
    self.gen_delay_min = GEN_DELAY_MIN
    self.gen_delay_amount = GEN_DELAY_AMOUNT
    self.gen_delay_avg = GEN_DELAY_AVG
    self.gen_delay = self.rand_delay()
    self.th0 = 300
    self.th1 = 140
    self.action_space = space(0)
    #self.observation_space = np.array(np.zeros((4,1)))
    self.observation_space = np.array((200,100,0,0))
    self.state = np.array([self.th0,self.th1,0,0])
    self.acc_avg = 150
  
  def reset(self,isRandomDelayMin):
    if isRandomDelayMin == 1:
      self.gen_delay_min = np.fix(np.random.normal(20,60))
      self.gen_delay = self.rand_delay()
    self.th0 = 300
    self.th1 = 140
    self.acc_avg = 150
    ob_avg = round(np.mean(self.gen_delay),0)
    ob_std = round(np.std(self.gen_delay) ,0)
    #self.state = [self.th0,self.th1,ob_avg,ob_std] # state 由 rand_delay的th0、th1、avg、var
    self.state = np.array([self.th0,self.th1,ob_avg,ob_std])
    #print("reset env, state: ",self.state)
    return self.state 

  def rand_delay(self):
    # 1. 产生截断泊松分布/截断高斯分布 的随机时延（组）
    # 2. 随机插入大时延
    scale_b = np.random.uniform(25,50) # 高斯抖动的标准差
    scale = 40
    a = np.zeros(self.gen_delay_amount)
    temp = np.fix(10*(np.random.poisson(lam = self.gen_delay_avg/20, size = self.gen_delay_amount)) + np.random.normal(0, scale_b,size = self.gen_delay_amount))
    for i in range(self.gen_delay_amount):
        a[i] = np.random.normal(0,scale) 
        if abs(a[i]) <= (scale*2):
          a[i] = 0
        elif abs(a[i] >= (scale*4)):
          a[i] = 0
        else:  
          a[i] = round(a[i],2)
    # np.sum([temp,a], axis = 0)
    temp += self.gen_delay_min
    return np.sum([temp,a], axis = 0)
    
  def calc_TAP(self):
    ''' simple TAP calculation '''
    a = 0
    k_final = 0
    k = 0
    acc_avg = 0
    count = 0
    TAP = 0 
    success = 1
    if self.th1 > self.th0:
      return 0, 0, -1
    while a < self.gen_delay_amount:           
        if abs(self.gen_delay[a]) >= self.th0:
          TAP = 0
          k = 0
          a += 1
          continue
        elif abs(self.gen_delay[a]) >= self.th1:
          a += 1
          k += 1
          continue
        else:
          k += 1
          TAP += self.gen_delay[a] 
          a += 1
        if (k>=80) and (abs(TAP/k) <= 500):
          count += 1
          k_final += k
          acc_avg += TAP/k
          TAP = 0
          k = 0 
        if count >= 20:
          break
    if count<20:
      return 0, 0, 0 
    try:
      k_final = round(k_final/count,2)
      acc_avg = round(acc_avg/count,2)
    finally:
      return k_final, acc_avg, success
  
  def step(self,action):
    next_state = self.state
    if action <= 2:
      self.th0 += 20*(action-1) # 0,1,2
      reward = -0.1 if action==2 else 0
    elif action <= 5:
      self.th1 += 20*(action-4) # 3,4,5
      reward = 0 if action==4 else -0.1
      #
    if self.th0 <= 20:
      self.th0 = 40
      reward += -0.1
    if self.th1 >= self.th0:
      self.th1 = self.th0 - 20
      reward += -0.1
    if self.th1 <= 10:
      self.th1 = 30
      reward += -0.5

    k_final, acc_avg, success = self.calc_TAP()
    info = acc_avg

    if success == 0:
      reward += -2
    elif success == -1:
      print('th0 < th1 ! This shouldn\'t happen!')
    else:
      #des_acc = min(((abs(self.acc_avg) - abs(acc_avg))/10), 1) + min(self.th1/200 , 0.8)
      des_acc = min(100/acc_avg  + self.th1/600 -0.5 , 1)
      des_acc = max(des_acc,  -2)
      reward += round( des_acc, 2 )
      
      self.acc_avg = acc_avg
      next_state[0] = self.th0
      next_state[1] = self.th1
    done = 1 if k_final >= 400 else 0
    #print('success:',success,'reward: ',reward,'acc_avg: ',acc_avg)
    self.draw_pic_reward_total.append(reward)
    return next_state, reward, done, info # observation(next_state), reward, done, info

  def render(self):
    print('[render]: ',self.state)
    return 1

# ---------------------------------------------------------
# Hyper Parameters

EPISODE = 3001 # Episode limitation
STEP = 300 # Step limitation in an episode
TEST = 10 # The number of experiment test every 100 episode

# Environment Parameters
GEN_DELAY_AMOUNT = 10000
GEN_DELAY_MIN = 40 # ns, 1us → 300m
GEN_DELAY_AVG = 200
RANDOM_ENV = 1

def main():
  # initialize env and dqn agent
  env = TAP_env(GEN_DELAY_AMOUNT, GEN_DELAY_MIN,GEN_DELAY_AVG)
  agent = DQN(env)

  for episode in range(EPISODE):
    # initialize task 每一步的时候重设环境
    state = env.reset(RANDOM_ENV)
    print('episode: ',episode)
    # Train
    for step in range(STEP):
      action = agent.egreedy_action(state) # e-greedy action for train
      next_state,reward,done,acc_now = env.step(action)

      # Define reward for agent
      if done:
        reward = -1  # else reward = xx.xx
      agent.perceive(state,action,reward,next_state,done)
      state = next_state
      if done:
        break  
      if acc_now >= 10: 
          env.draw_pic_acc_total.append(acc_now)
    print('state of this episode: ',env.state,'acc of this episode at last: ',acc_now)
    # Test every 100 episodes
    if episode % 100 == 0:
      total_reward = 0
      total_info = 0
      for i in range(TEST):
        state = env.reset(RANDOM_ENV)
        for j in range(STEP):
          #env.render()
          action = agent.action(state) # direct action for test
          state,reward,done,info = env.step(action)
          total_reward += reward 
          total_info += info
          if done:
            break
        if info >= 10: 
          env.draw_pic_acc_total_test.append(info)  
      ave_reward = total_reward/TEST
      total_info = total_info/TEST/STEP
      print ('episode: ',episode,'Evaluation Average Reward:',ave_reward,'acc_at_last: ',total_info)
      print ('state: ',state,'Q_value: ',agent.Q_value_save)
  
  #--Output to File------------------------------------------#
  fileObject = open('reward.txt', 'w')
  for i in env.draw_pic_reward_total:
    fileObject.write(str(i)) 
    fileObject.write('\n') 
  fileObject.close()
  
  fileObject = open('acc.txt', 'w')
  for i in env.draw_pic_acc_total:
    fileObject.write(str(i)) 
    fileObject.write('\n') 
  fileObject.close()

  agent.saver.save(agent.session,"model_save")



if __name__ == '__main__':
  main()


