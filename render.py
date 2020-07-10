import matplotlib.pyplot as plt
import gym
from gym import wrappers
import random
import collections
import numpy as np
import tensorflow as tf
from dqn import Qnet
import time
################################################################
# suppress tensorflow logs. 
# Solution from https://stackoverflow.com/a/57497465
import logging, os 
logging.disable(logging.WARNING) 
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
################################################################

# def record_video(q , n_epi, vid_dir, frame_number):
#     rec_env = wrappers.Monitor(rec_env, "/tmp/CartPole-v1", video_callable = False, force = True)
#     s = rec_env.reset()
#     done = False
#     plt.imshow(rec_env.render(mode='rgb_array'))
#     plt.title("DQN-CartPole. | Episode: %d" % (n_epi))
#     plt.axis('off')        
#     plt.savefig(vid_dir+"image"+str(frame_number)+".png")
    
#     while not done:
#         a = np.argmax(q.model.predict(tf.constant(s,shape=(1,input_shape))))
#         s_prime, _ , done, _ = rec_env.step(a)
#         s = s_prime
#         frame_number += 1
#         plt.imshow(rec_env.render(mode='rgb_array'))
#         plt.title("DQN-CartPole. | Episode: %d" % (n_epi))
#         plt.axis('off')  
#         plt.savefig(vid_dir+"image"+str(frame_number)+".png")
#         if done:
#             rec_env.close()
#             break
#     return frame_number

######## hyperparameters###################
BUFFERLIMIT = 50_000
MINI_BATCH_SIZE = 32 
HIDDEN_LAYERS = 2
HIDDEN_LAYER_UNITS = 64
LEARNING_RATE = 0.0005
DISCOUNT_RATE  = 0.99 
EPISODES = 40 # total nusmber of episodes to train for
UPDATE_TARGET_INTERVAL = 100  # target update interval for hard update
TAU = 0.0001 # used when soft update is used
############################################

seed_val = 0
env = gym.make("CartPole-v1") # select environment. Currently only tested on CartPole-v1
env.seed(seed_val)
np.random.seed(seed_val)
tf.random.set_seed(seed_val)
random.seed(seed_val)

log_dir = "final_experiment" # directory name to store log data
exp_name = "final_experiment" # name of the experiment
savename = exp_name
model_dir = "Model/"+log_dir+"/"
if not os.path.exists(model_dir):
    os.mkdir(model_dir)    
model_dir = model_dir+"DQN_"+savename

vid_dir = "Video/"+log_dir+"/"
if not os.path.exists(vid_dir):
    os.mkdir(vid_dir)
save_vid_cmd = "ffmpeg -y -framerate 12 -i "+vid_dir+"image%d.png -c:v libx264 -profile:v high -crf 20 -pix_fmt yuv420p progress.mp4"

input_shape = env.observation_space.shape[0] # input shape for the neural net
output_shape = env.action_space.n # output shape for the neural net
frame_number = 0

q = Qnet(
    hidden_layers = HIDDEN_LAYERS, 
    observation_space = input_shape, 
    action_space = output_shape, 
    learning_rate = LEARNING_RATE, 
    units = HIDDEN_LAYER_UNITS
    ) # the main Q-network

for i in range(0,41,10):
    print(i)
    ep_model =  model_dir+"_"+str(i)
    q.model.load_weights(ep_model+"/"+"DQN_"+savename+"_"+str(i))
    s = env.reset()
    score = 0.
    done = False
    plt.imshow(env.render(mode='rgb_array'))
    plt.title("DQN-CartPole. | Episode: %d" % (i))
    plt.axis('off')        
    plt.savefig(vid_dir+"image"+str(frame_number)+".png")
    ##############################################################
    while not done:
        a = np.argmax(q.model.predict(tf.constant(s,shape=(1,input_shape))))
        s_prime, r , done, _ = env.step(a)
        s = s_prime
        score += r
        frame_number += 1
        plt.imshow(env.render(mode='rgb_array'))
        plt.title("DQN-CartPole. | Episode: %d" % (i))
        plt.axis('off')  
        plt.savefig(vid_dir+"image"+str(frame_number)+".png")
        time.sleep(1/120)
        if done:
            env.close()
            break
    print("Episode: {}. Rewards: {}".format(i,score))
os.system(save_vid_cmd)