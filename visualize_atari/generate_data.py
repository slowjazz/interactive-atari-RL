#!/usr/bin/env python
# coding: utf-8

# # Jacobian vs. Perturbation
# Visualizing and Understanding Atari Agents | Sam Greydanus | 2017 | MIT License

# In[2]:


from __future__ import print_function
import warnings ; warnings.filterwarnings('ignore') # mute warnings, live dangerously

import pandas as pd
import h5py 

import torch
from torch.autograd import Variable
import torch.nn.functional as F

import gym, os, sys, time, argparse
sys.path.append('..')
from visualize_atari import *


env_name = 'Breakout-v0'
save_dir = 'figures/'
load_dir = '../baby-a3c/breakout-v4/models_model7-02-17-20-41/'
def get_rollout(it, seed = 1):
    print("set up dir variables and environment...")
    #load_dir = '{}/'.format(env_name.lower())
    
    meta = get_env_meta(env_name)
    env = gym.make(env_name) ; env.seed(seed)

    print("initialize agent and try to load saved weights...")
    model = NNPolicy(channels=1, num_actions=env.action_space.n, memsize=256)
    _ = model.try_load(load_dir, checkpoint='*'+str(it)+'.tar') ; torch.manual_seed(1)
    
    print("get a rollout of the policy...")
    history = rollout(model, env, max_ep_len=3e3)
    return model, history



if 'model_rollouts_5.h5' not in os.listdir('../static'):
    store = h5py.File('../static/model_rollouts_5.h5','w')
else:
    store = h5py.File('../static/model_rollouts_5.h5','w')



exceptions = []
episodes = 5 # episodes to replay per iteration
iterations = [1,19,30,40,50,60,70,80,90,100] # iterations we want to look at
#iterations=[30]

def filter_iterations(x):
    return any('.'+str(i)+'.' in x for i in iterations)

for modelname in filter(filter_iterations, os.listdir(load_dir)):
    iteration = modelname.split('.')[-2]
    print(modelname)
    for ep in range(episodes):
#         try:
#             _, history = get_rollout(iteration, ep+1)
#         except:
#             exceptions.append(iteration)
#             continue
        _, history = get_rollout(iteration, ep+1)
        path = 'models_model7-02-17-20-41'
        for k in history.keys():
            target = np.stack(history[k], axis=0)[::5]
            store.create_dataset(os.path.join(path, modelname, 
                                'history',str(ep), k), data = target,
                                compression = "gzip")
            print('saved rollout at', os.path.join(path, modelname, 
                                'history',str(ep), k))
        
#exceptions


# # store saliency maps

# In[5]:


def saliency_on_atari_frame_short(saliency, atari, fudge_factor, channel=2, sigma=0):
    # sometimes saliency maps are a bit clearer if you blur them
    # slightly...sigma adjusts the radius of that blur
    pmax = saliency.max()
    S = imresize(saliency, size=[160,160], interp='bilinear').astype(np.float32)
    S = S if sigma == 0 else gaussian_filter(S, sigma=sigma)
    S -= S.min() ; S = fudge_factor*pmax * S / S.max()
    return S
#     I = atari.astype('uint16')
#     I[35:195,:,channel] += S.astype('uint16')
#     I = I.clip(1,255).astype('uint8')
#     return I

for iteration in [1,19,30,40,50,60,70,80,90,100]:
    print("running iteration ",iteration)
    
    
    for ep in range(episodes):
        ins = store['models_model7-02-17-20-41/model.'+str(iteration)+'.tar/history/'+str(ep)+'/ins'].value
        hx = store['models_model7-02-17-20-41/model.'+str(iteration)+'.tar/history/'+str(ep)+'/hx'].value
        history = {'ins':ins, 'hx':hx}

        meta = get_env_meta(env_name)
        env = gym.make(env_name) ; env.seed(1)
        model = NNPolicy(channels=1, num_actions=env.action_space.n, memsize=256)
        try:
            _ = model.try_load(load_dir, checkpoint='*.'+str(iteration)+'.tar') ; torch.manual_seed(ep+1)
        except:
            print("exception at iteration: ",iteration)
            continue
        
        radius = 5
        density = 5
        actor_frames = []
        critic_frames = []
        # 5 frame increments stored in rollout data, we use 5 frame increments to generate saliency maps
        for frame_ix in range(0, hx.shape[0]):
            if frame_ix >= len(history['ins']):
                break
            actor_saliency = score_frame(model, history, frame_ix, radius, density, interp_func=occlude, mode='actor')
            critic_saliency = score_frame(model, history, frame_ix, radius, density, interp_func=occlude, mode='critic')
            frame = history['ins'][frame_ix].squeeze().copy()
            
            actor_map = saliency_on_atari_frame_short(actor_saliency, frame, fudge_factor=100, channel=2)
            critic_map = saliency_on_atari_frame_short(critic_saliency, frame, fudge_factor=1000, channel=0)
            
            actor_frames.append(actor_map)
            critic_frames.append(critic_map)
            print("saved {}th frame & sal-map", frame_ix)
            
        actor_frames = np.array(actor_frames)
        critic_frames = np.array(critic_frames)
        path = 'models_model7-02-17-20-41'
        modelname = 'model.'+str(iteration)+'.tar'
        actor_path = os.path.join(path, modelname, 'history',str(ep), 'actor_sal')
        critic_path = os.path.join(path, modelname, 'history',str(ep), 'critic_sal')
        if actor_path in store:
            store[actor_path] = actor_frames
            store[critic_path] = critic_frames
        else:
            store.create_dataset(actor_path, data = actor_frames) 
            store.create_dataset(critic_path, data = critic_frames)
        print('saved saliency at', os.path.join(path, modelname, 
                                'history',str(ep), '<SAL_TYPE>'))
        
