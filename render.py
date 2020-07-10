import matplotlib.pyplot as plt
import matplotlib.animation as animation
import gym
import random
import numpy as np
import tensorflow as tf
from dqn import Qnet

################################################################
# suppress tensorflow logs. 
# Solution from https://stackoverflow.com/a/57497465
import logging, os 
logging.disable(logging.WARNING) 
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
################################################################
Writer = animation.writers['ffmpeg']
writer = Writer(fps=120, metadata=dict(artist='Me'), bitrate=1800)
fig = plt.figure()
ax = fig.add_subplot(111)
ims = []
######## hyperparameters###################
# Set up formatting for the movie files
HIDDEN_LAYERS = 2
HIDDEN_LAYER_UNITS = 64
LEARNING_RATE = 0.0005
DISCOUNT_RATE  = 0.99 
EPISODES = 40 # total nusmber of episodes to train for
############################################
seed_val = 0
env = gym.make("CartPole-v1") # select environment. Currently only tested on CartPole-v1
###############################
env.seed(seed_val)
np.random.seed(seed_val)
tf.random.set_seed(seed_val)
random.seed(seed_val)
################################
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

################################
input_shape = env.observation_space.shape[0] # input shape for the neural net
output_shape = env.action_space.n # output shape for the neural net
q = Qnet(
    hidden_layers = HIDDEN_LAYERS, 
    observation_space = input_shape, 
    action_space = output_shape, 
    learning_rate = LEARNING_RATE, 
    units = HIDDEN_LAYER_UNITS
    ) # the main Q-network
################################
for i in range(0,EPISODES+1,10):
    ep_model =  model_dir+"_"+str(i)
    q.model.load_weights(ep_model+"/"+"DQN_"+savename+"_"+str(i))
    s = env.reset()
    score = 0.
    done = False
    im = plt.imshow(env.render(mode='rgb_array'))
    ttl = plt.text(0.5,1.01,"DQN-CartPole-v1. | Episode: "+str(i), horizontalalignment = 'center', verticalalignment = 'bottom', transform = ax.transAxes)
    plt.axis('off')
    ims.append([im, ttl])  

    ##############################################################
    while not done:
        a = np.argmax(q.model.predict(tf.constant(s,shape=(1,input_shape))))
        s_prime, r , done, _ = env.step(a)
        s = s_prime
        score += r
        im=plt.imshow(env.render(mode='rgb_array'))
        ttl = plt.text(0.5,1.01,"DQN-CartPole-v1. | Episode: "+str(i), horizontalalignment = 'center', verticalalignment = 'bottom', transform = ax.transAxes)
        plt.axis('off')  
        ims.append([im, ttl])
        #plt.savefig(vid_dir+"image"+str(frame_number)+".png")
        #time.sleep(1/120)
        if done:
            # env.close()
            break
    print("Episode: {}. Rewards: {}".format(i,score))

ani = animation.ArtistAnimation(fig, ims, interval=80, blit=True)
ani.save(vid_dir+'progress.mp4')