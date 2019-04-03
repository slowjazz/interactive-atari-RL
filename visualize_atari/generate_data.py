#!/usr/bin/env python
# coding: utf-8

# # Jacobian vs. Perturbation
# Visualizing and Understanding Atari Agents | Sam Greydanus | 2017 | MIT License

# Arg scheme
# 0 for only rollouts, 1 for only saliency, none for both

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
model_folder = 'models_model7-02-17-20-41'
load_dir = '../baby-a3c/breakout-v4/'+model_folder + '/'
def get_rollout(model, seed = 1):
    meta = get_env_meta(env_name)
    env = gym.make(env_name) ; env.seed(seed)
    env.reset()
    history = rollout(model, env, max_ep_len=3e3)
    env.close()
    return history



if 'model_rollouts_5.h5' not in os.listdir('../static'):
    store = h5py.File('../static/model_rollouts_5.h5','w')
else:
    store = h5py.File('../static/model_rollouts_5.h5','a')



exceptions = []
episodes = 5 # episodes to replay per iteration
iterations = [1,19,30,40,50,60,70,80,90,100] # iterations we want to look at
#iterations=[30]

def filter_iterations(x):
    return any('.'+str(i)+'.' in x for i in iterations)

if len(sys.argv)==1 or sys.argv[1]==1:
    for modelname in filter(filter_iterations, os.listdir(load_dir)):
        iteration = modelname.split('.')[-2]
        print(modelname)
        print("initialize agent and try to load saved weights...")
        model = NNPolicy(channels=1, num_actions=4, memsize=256)
        _ = model.try_load(load_dir, checkpoint='*'+str(iteration)+'.tar')
        torch.manual_seed(1)
        for ep in range(episodes):
            print(f'getting rollout of {modelname} on ep {ep}')
            history = get_rollout(model, ep + 1)
            path = model_folder
            for k in history.keys():
                target = np.stack(history[k], axis=0)
                if k == "ins":
                    store.create_dataset(os.path.join(path, modelname, 
                                    'history',str(ep), k), data = target,
                                    compression = "gzip")
                else:
                    store.create_dataset(os.path.join(path, modelname, 
                    'history',str(ep), k), data = target)
                print('saved rollout at', os.path.join(path, modelname, 
                                    'history',str(ep), k))

# do only saliency
if len(sys.argv) > 1 and sys.argv[1]==1:
    store.close()
    exit()
    
from torch.multiprocessing import Pool
CPU_COUNT = torch.multiprocessing.cpu_count()

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

def run_through_model(model, ins_instance, hx_instance, interp_func=None, mask=None, blur_memory=None, mode='actor'):
    if mask is None:
        im = prepro(ins_instance)
    else:
        assert(interp_func is not None, "interp func cannot be none")
        im = interp_func(prepro(ins_instance).squeeze(), mask).reshape(1,80,80) # perturb input I -> I'
    tens_state = torch.Tensor(im)
    state = Variable(tens_state.unsqueeze(0), volatile=True)
    hx = Variable(torch.Tensor(hx_instance).view(1,-1))
    #hx = Variable(torch.Tensor(history['hx'][ix-1]).view(1,-1))
    #cx = Variable(torch.Tensor(history['cx'][ix-1]).view(1,-1))
    #if blur_memory is not None: cx.mul_(1-blur_memory) # perturb memory vector
    #return model((state, (hx, cx)))[0] if mode == 'critic' else model((state, (hx, cx)))[1]
    return model((state, hx))[0] if mode == 'critic' else model((state, hx))[1]

def score_frame(model, ins_instance, hx_instance, r, d, interp_func, mode='actor'):
    # r: radius of blur
    # d: density of scores (if d==1, then get a score for every pixel...
    #    if d==2 then every other, which is 25% of total pixels for a 2D image)
    assert mode in ['actor', 'critic'], 'mode must be either "actor" or "critic"'
    L = run_through_model(model, ins_instance, hx_instance, interp_func, mask=None, mode=mode)
    scores = np.zeros((int(80/d)+1,int(80/d)+1)) # saliency scores S(t,i,j)
    for i in range(0,80,d):
        for j in range(0,80,d):
            mask = get_mask(center=[i,j], size=[80,80], r=r)
            l = run_through_model(model, ins_instance, hx_instance, interp_func, mask=mask, mode=mode)
            #scores[int(i/d),int(j/d)] = (L-l).pow(2).sum().mul_(.5).data[0]
            scores[int(i/d),int(j/d)] = (L-l).pow(2).sum().mul_(.5).item()
    pmax = scores.max()
    scores = imresize(scores, size=[80,80], interp='bilinear').astype(np.float32)
    print('done scoring frame')
    return pmax * scores / scores.max()

# prepro = lambda img: imresize(img[35:195].mean(2), (80,80)).astype(np.float32).reshape(1,80,80)/255.
# searchlight = lambda I, mask: I*mask + gaussian_filter(I, sigma=3)*(1-mask) # choose an area NOT to blur
# occlude = lambda I, mask: I*(1-mask) + gaussian_filter(I, sigma=3)*mask # choose an area to blur


def get_saliency_mp(model, ins_instance, hx_instance):
    radius = 5
    density = 5
    print(ins_instance.shape, hx_instance.shape)
    actor_saliency = score_frame(model, ins_instance, hx_instance, radius, density, interp_func=occlude, mode='actor')
    critic_saliency = score_frame(model, ins_instance, hx_instance, radius, density, interp_func=occlude, mode='critic')
    frame = ins_instance.squeeze().copy()

    actor_map = saliency_on_atari_frame_short(actor_saliency, frame, fudge_factor=100, channel=2)
    critic_map = saliency_on_atari_frame_short(critic_saliency, frame, fudge_factor=1000, channel=0)

    return (actor_map, critic_map)

pool = Pool(processes = CPU_COUNT)
for iteration in [1,19,30,40,50,60,70,80,90,100]:
    print("running iteration ",iteration)
    try:
        model = NNPolicy(channels=1, num_actions=4, memsize=256)
        _ = model.try_load(load_dir, checkpoint='*.'+str(iteration)+'.tar')
        
    except:
        print("exception at iteration: ",iteration)
        continue
    
    for ep in range(episodes):
        print(f'getting saliency maps of {iteration} on ep {ep}')
        ins = store[model_folder + '/model.'+str(iteration)+'.tar/history/'+str(ep)+'/ins'].value[::5] # 5 frame increments
        hx = store[model_folder + '/model.'+str(iteration)+'.tar/history/'+str(ep)+'/hx'].value[::5] # 5 frame increments
        history = {'ins':ins, 'hx':hx}

        meta = get_env_meta(env_name)
        env = gym.make(env_name) ; env.seed(ep+1)
        env.reset()
        
#         radius = 5
#         density = 5
#         actor_frames = []
#         critic_frames = []
        frame_data = ins.copy()
        
        # 1 frame increments stored in rollout data, we use 5 frame increments to generate saliency maps
#         for frame_ix in range(0, ins.shape[0], 5):
#             actor_saliency = score_frame(model, history, frame_ix, radius, density, interp_func=occlude, mode='actor')
#             critic_saliency = score_frame(model, history, frame_ix, radius, density, interp_func=occlude, mode='critic')
#             frame = history['ins'][frame_ix].squeeze().copy()
            
#             actor_map = saliency_on_atari_frame_short(actor_saliency, frame, fudge_factor=100, channel=2)
#             critic_map = saliency_on_atari_frame_short(critic_saliency, frame, fudge_factor=1000, channel=0)
            
#             actor_frames.append(actor_map)
#             critic_frames.append(critic_map)
            
#         actor_frames = np.array(actor_frames)
#         critic_frames = np.array(critic_frames)
        
        model_rep = [model for i in range(len(frame_data))]
        mp_in = zip(model_rep, frame_data, np.roll(hx, 1, 0))
        actor_frames, critic_frames = zip(*pool.starmap(get_saliency_mp, mp_in))

        path = model_folder
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

Pool.close()
Pool.join()

store.close()
        
