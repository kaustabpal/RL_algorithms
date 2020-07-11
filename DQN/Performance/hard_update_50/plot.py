from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd
from operator import add
from operator import sub
#sns.set()

orange = "#fe8112"
plot_title = "DQN-CartPole. Hard Update. Update interval = 50"
beta = 0.88
TAU0_ = "0001"
MINI_BATCH_SIZE = [32, 64, 128, 256]
MINI_BATCH_COL = ['#f700ff', '#0000ff', '#00ff44', '#ffa600']
col_names = ['Episode', 'mean_score', 'std_dev']
save_plot = "hard_update_50.png"
for i in range(len(MINI_BATCH_SIZE)):
    read_performance="hard_update_50_minibatch_"+str(MINI_BATCH_SIZE[i])+".csv"
    data = pd.read_csv(read_performance, names=col_names)
    episode = data.Episode.tolist()[1:]
    episode[:]=[int(e) for e in episode]
    score = data.mean_score.tolist()[1:]
    score[:]=[float(s) for s in score]
    for j in range(1,len(score)):
        score[j] = beta*score[j-1] + (1-beta)*score[j]
    std = data.std_dev.tolist()[1:]
    std[:]=[float(std) for std in std]
    for j in range(1,len(std)):
        std[j] = beta*std[j-1] + (1-beta)*std[j]
    y_max = list(map(add, score, std))
    y_min = list(map(sub, score, std))
    plt.ylim((0,500))
    plt.xlim((0,990))
    plt.xlabel('Episodes')
    plt.ylabel('Rewards')
    plt.title(plot_title)
    plt.plot(episode,score, label="MINI_BATCH_SIZE ="+str(MINI_BATCH_SIZE[i])) #, color = MINI_BATCH_COL[i])
    plt.fill_between(episode, y_min, y_max, alpha=0.1)
#############################################################################
plt.vlines([50,100,150,200,250,300,350,400,450,500,550,600,650,700,750,800,850,900],0,500,color ='k', linestyles = 'dotted')
plt.legend(loc="upper left")
#plt.show()
plt.savefig(save_plot)