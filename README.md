# Interactive Visual Reinforcement Learning on Atari's Breakout
#### A CS-8395 Project (WIP)

This project intends to address challenges with understanding complex reinforcement learning agents on complicated tasks like Atari games through visual aids.  

### Update 1 

**Tasks done:**
- Train A3C agent, amend logging for custom use
- Change code of saliency map visualization for new model (LSTM vs. GRU)
- Generate sample saliency maps 
- Put up the Dashboard, learn some Dash/plotly

**Instructions:**
To run the dashboard:
```
python app.py # localhost:8050
```

To train a new model:
```
cd baby-a3c
python baby-a3c.py --load_model <model name> 
```

To get saliency maps:
```
jupyter notebook
-> visualize_atari/jacobian-vs-perturbation.ipynb
```

**Environment:**
- Python 3.6
- PyTorch 1.0 
- [Dash](https://dash.plot.ly/installation)

# References/Source Material
Code adapted from Sam Greydanus' work:

https://github.com/greydanus/visualize_atari - generate saliency maps of agent playthroughs
https://github.com/greydanus/baby-a3c - for a3c model training 
https://arxiv.org/abs/1711.00138 

MIT License
