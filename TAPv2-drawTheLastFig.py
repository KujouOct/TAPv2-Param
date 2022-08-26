import tensorflow as tf
import numpy as np
import random
from collections import deque


class space():
  def __init__(self,space_type):
    if space_type == 0:
      self.the_action_space = np.array((0,1,2,3,4,5))
      self.n = len(self.the_action_space)

class TAP_env():

  def __init__(self, GEN_DELAY_AMOUNT,GEN_DELAY_MIN,GEN_DELAY_AVG):
    print("Initializing env...")
    self.draw_pic_reward_total = deque()
    self.draw_pic_acc_total = deque()
    self.draw_pic_acc_total_test = deque()
    self.gen_delay_min = GEN_DELAY_MIN
    self.gen_delay_amount = GEN_DELAY_AMOUNT
    self.gen_delay_avg = GEN_DELAY_AVG
    self.gen_delay = self.rand_delay()
    self.th0 = 500
    self.th1 = 250
    self.action_space = space(0)
    #self.observation_space = np.array(np.zeros((4,1)))
    self.observation_space = np.array((200,100,0,0))
    self.state = np.array([self.th0,self.th1,0,0])
    self.acc_avg = 150
  
  def reset(self,isRandomDelayMin):
    if isRandomDelayMin == 1:
      self.gen_delay_min = np.fix(np.random.normal(20,60))
      self.gen_delay = self.rand_delay()
    self.th0 = 500
    self.th1 = 250
    self.acc_avg = 150
    ob_avg = round(np.mean(self.gen_delay),0)
    ob_std = round(np.std(self.gen_delay) ,0)
    self.state = np.array([self.th0,self.th1,ob_avg,ob_std])
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
      return 0, self.gen_delay[1], 0 
    try:
      k_final = round(k_final/count,2)
      acc_avg = round(acc_avg/count,2)
    finally:
      return k_final, acc_avg, success
  
  
# ---------------------------------------------------------

STEP = 300 # Step limitation in an episode

# Environment Parameters
GEN_DELAY_AMOUNT = 10000
GEN_DELAY_MIN = 40 # ns, 1us → 300m
GEN_DELAY_AVG = 200
RANDOM_ENV = 1

def main():
  # initialize env and dqn agent
  env = TAP_env(GEN_DELAY_AMOUNT, GEN_DELAY_MIN,GEN_DELAY_AVG)

  for episode in range(STEP):
    state = env.reset(RANDOM_ENV)
    _, acc_now, _ = env.calc_TAP()
    if acc_now >= 10:
      env.draw_pic_acc_total.append(acc_now)
  #--Output to File------------------------------------------#
  
  fileObject = open('last_fig_acc.txt', 'w')
  for i in env.draw_pic_acc_total:
    fileObject.write(str(i)) 
    fileObject.write('\n') 
  fileObject.close()


if __name__ == '__main__':
  main()


