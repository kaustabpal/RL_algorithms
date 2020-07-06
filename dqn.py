import time
import timeit
import math
import matplotlib.pyplot as plt
import pandas as pd
import gym
from gym import wrappers
import random
import collections
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from operator import add
from operator import sub
################################################################
# suppress tensorflow logs. 
# Solution from https://stackoverflow.com/a/57497465
import logging, os 
logging.disable(logging.WARNING) 
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
################################################################

class ReplayBuffer:
    def __init__(self):
        self.buffer = collections.deque(maxlen=BUFFERLIMIT)

    def put(self, transition):
        if(self.size() >= BUFFERLIMIT):
            self.buffer.pop()
        self.buffer.append(transition)

    def sample(self, mini_batch_size=32):
        mini_batch = random.sample(self.buffer, min(len(self.buffer), mini_batch_size))
        return mini_batch
    
    def size(self):
        return len(self.buffer)
######################################################################

class Qnet:
    def __init__(self, observation_space, action_space, learning_rate, h1 = 64, h2 = 64):
        self.input_shape = observation_space
        self.output_shape = action_space
        self.learning_rate = learning_rate
        self.h1 = h1
        self.h2 = h2
        self.model = tf.keras.models.Sequential([
            tf.keras.layers.Dense(self.h1, input_shape=(self.input_shape,), activation="relu"),
            tf.keras.layers.Dense(self.h2, activation="relu"),
            tf.keras.layers.Dense(self.output_shape, activation = "linear")         
        ])
        self.model.compile(optimizer =tf.keras.optimizers.Adam(lr=self.learning_rate), loss="mse")
    
    def train(self,q_target,replay,mini_batch_size=32):
        mini_batch = replay.sample(mini_batch_size)
        x = []
        y = []
        for s,a,r,s_prime,done in mini_batch:
            max_future_q = np.amax(q_target.model.predict(tf.constant(s_prime,shape=(1,self.input_shape))))
            target = r + DISCOUNT_RATE*max_future_q*np.invert(done)
            current_q = self.model.predict(tf.constant(s,shape=(1,self.input_shape))) # current q_values for the actions
            current_q[0][a] = target # updating the q_value of the chosen action to that of the target q value
            x.append(s)
            y.append(current_q)
        x = tf.constant(x,shape=(len(x), self.input_shape))
        y = tf.constant(y, shape=(len(y), self.output_shape))
        self.model.fit(x,y)

    def update_weight(self, q, ep_num, update_interval = 20, tau = 0.01, soft=False):
        if soft == False:
            # hard update
            if(ep_num % update_interval == 0):
                target_theta = q.model.get_weights()
                self.model.set_weights(target_theta)
                print("Target Update")
        else:
            # soft update
            q_theta = q.model.get_weights()
            target_theta = self.model.get_weights()
            counter = 0
            for q_weight, target_weight in zip(q_theta,target_theta):
                target_weight = target_weight * (1-tau) + q_weight * tau
                target_theta[counter] = target_weight
                counter += 1
            self.model.set_weights(target_theta)
    
    def sample_action(self, obs, epsilon):
        coin = random.random()
        if(coin<=epsilon):
            return random.randint(0, 1) # returns values between 0 and 1
        else:
            return np.argmax(self.model.predict(obs))
######################################################################

def test(n_epi, epsilon, q, seed_val):
    rwrd = []
    test_env = gym.make("CartPole-v1")
    test_env.seed(seed_val) # setting seed for the test_env
    for eps in range(10):        
        test_s = test_env.reset()
        test_done = False
        test_score = 0
        while not test_done:
            test_a = np.argmax(q.model.predict(tf.constant(test_s,shape=(1,input_shape))))
            test_s_prime, test_r, test_done, test_info =test_env.step(test_a)
            test_s = test_s_prime
            test_score += test_r # we get +1 reward every timestep
            if test_done:
                rwrd.append(test_score)
                break
    mean_score = np.mean(rwrd)
    std_score = round(np.std(rwrd),3)
    print("Episode: {}. Score: {}. Std: {}. Epsilon: {}".format(n_epi, mean_score, std_score, round(epsilon,3)))
    return mean_score, std_score  

if __name__=="__main__":

    seed_val = 0
    np.random.seed(seed_val)
    tf.random.set_seed(seed_val)
    random.seed(seed_val)

    # hyperparameters
    BUFFERLIMIT = 50_000
    MINI_BATCH_SIZE = [32, 64, 128, 256]
    H_1 = 64
    H_2 = 64
    LEARNING_RATE = 0.0005
    LEARNING_RATE_DECAY = 0.00025 
    DISCOUNT_RATE  = 0.99 
    EPISODES = 1000 # total nusmber of episodes to train for
    soft_update = False
    # for future experiments, only change these three values
    UPDATE_TARGET_INTERVAL = 20  # Used when hard update is used 
    TAU = 0.0001 # used when soft update is used
    target_dir = "hard_update_20" # hard_update_20 50 100 200

    temp_env = gym.make("CartPole-v1")

    input_shape = temp_env.observation_space.shape[0]
    output_shape = temp_env.action_space.n

    temp_q = Qnet(input_shape,output_shape, LEARNING_RATE, h1= H_1, h2 = H_2)
    initial_weights = temp_q.model.get_weights()
    del temp_q
    del temp_env

    start_time = timeit.default_timer()

    for i in range(len(MINI_BATCH_SIZE)):
        env = gym.make("CartPole-v1") # select environment. Currently only tested on CartPole-v1
        env.seed(seed_val)  # setting seed for the env

        q = Qnet(input_shape,output_shape, LEARNING_RATE, h1= H_1, h2 = H_2)
        q_target = Qnet(input_shape, output_shape, LEARNING_RATE, h1= H_1, h2 = H_2)
        q.model.set_weights(initial_weights)
        q_target.model.set_weights(initial_weights)
        memory = ReplayBuffer()

        ep_vec = [] # Vector to store ep_number for plotting
        mean_score_vec = [] # vector to store score in this episode for plotting
        std_vec =[]

        savename = target_dir+"_minibatch_"+str(MINI_BATCH_SIZE[i])
        save_performance = "Performance/"+target_dir+"/"+savename+".csv"
        save_plot = "Performance/"+target_dir+"/"+savename+".png"
        save_model = "Model/"+target_dir+"/DQN_"+savename
        plot_title = "DQN-CartPole. Hard update. Update interval = 20."

        print("\n**************** Starting experiment on minibatch size: {} ****************\n".format(MINI_BATCH_SIZE[i]))

        ##################################################################
        for n_epi in range(EPISODES):
            epsilon = max(0.01, (0.99 - 0.98/200*n_epi))
            s = env.reset()
            done = False
            score = 0.
            ##############################################################
            while not done:
                a = q.sample_action(tf.constant(s,shape=(1,input_shape)), epsilon) #select action from updated q net
                s_prime, r, done, info = env.step(a)
                memory.put((s,a,r,s_prime,int(done))) # insert into experience replay
                s = s_prime
                score += r
                if done:
                    break
            ###############################################################        
            if(memory.size() >= 1000):
                q.train(q_target, memory, MINI_BATCH_SIZE[i])    # update q net
                q_target.update_weight(q, ep_num=n_epi, update_interval=UPDATE_TARGET_INTERVAL, tau=TAU, soft=soft_update)
            if(n_epi % 10 ==0):
                ep_vec.append(n_epi)
                mean_, std_ =test(n_epi, epsilon, q, seed_val)
                mean_score_vec.append(mean_)
                std_vec.append(std_)        
        ####################################################################   

        ##### plot showing score vs episodes #######
        y_max=list(map(add, mean_score_vec, std_vec))
        y_min=list(map(sub, mean_score_vec, std_vec))
        plt.ylim((0,500))
        plt.xlim((0,EPISODES-10))
        plt.xlabel('Episodes')
        plt.ylabel('Rewards')
        plt.title(plot_title)
        plt.plot(ep_vec,mean_score_vec)
        plt.fill_between(ep_vec, y_min, y_max, alpha=0.1)
        
        ########################################

        ############ SAVING DATAS ###########################
        performance_data = {'Episode':ep_vec, 'mean_score': mean_score_vec, 'std_dev': std_vec}
        df = pd.DataFrame(performance_data) 
        df.to_csv(save_performance, index=False)

        q.model.save_weights(save_model)

        del q
        del q_target
        del memory
        del env

    plt.savefig(save_plot)
    stop_time = timeit.default_timer()
    print("TIME TAKEN: {}".format(stop_time-start_time))
