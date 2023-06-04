import torch
import numpy as np
import gym
import os
import random
import torch.optim as optim
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import AgentConfig, EnvConfig
# from memory import ReplayMemory
from network_rnn import Encoder_rnn, Decoder_rnn, Seq2Seq
from rnn_autoencoder import RNNAutoencoder
# from ops import *


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

env = gym.make('Pong-ram-v0')
# s = env.reset()
# done = False

# for i in range(10):
#     a = env.action_space.sample()
#     print("a : ", a)
#     print(env.step(a))

INPUT_DIM = 130
OUTPUT_DIM = 130
HID_DIM = 512
N_LAYERS = 2
ENC_DROPOUT = 0
DEC_DROPOUT = 0
learning_rate = 0.001

# enc = Encoder_rnn(INPUT_DIM, HID_DIM, N_LAYERS, ENC_DROPOUT)
# dec = Decoder_rnn(OUTPUT_DIM, HID_DIM, N_LAYERS, DEC_DROPOUT)

# model = Seq2Seq(enc, dec, device).to(device)

# optimizer = optim.Adam(model.parameters())
# criterion = nn.MSELoss()

# You would need to define how long a sequence you want.
# For example, let's say you want sequences of length 10.
SEQUENCE_LENGTH = 100

def train(autoencoder, env, optimizer, criterion, n_steps):
    autoencoder.train()
    # episode_data = []
    
    
    for step in range(n_steps):
        if step == 0 or done:
            print("new!")
            state = env.reset()
            state = state[0]
            # print("state1 :",state)
            # episode_data = []  # Reset the episode data
            episode_data = torch.empty(0, 130).to(device) 
        
        action = env.action_space.sample()
        next_state, reward, done, trun, info = env.step(action)
        # print("state2 :",next_state)

        state = np.array(state).flatten()/255
        # print("werw", state)
        single_data = torch.tensor(np.concatenate([state, [action, reward]]), dtype=torch.float32).to(device)
        # episode_data.append(single_data)
        # print(episode_data.shape, single_data.shape)
        episode_data = torch.cat((episode_data, single_data.unsqueeze(0)), dim=0) 

        # If we have enough data for a sequence, train on that sequence
        if step % SEQUENCE_LENGTH == 0 and step > 0:
            # Convert the episode data into a proper tensor
            sequence_data = episode_data  # Add batch dimension
            
            
            # Reset gradients
            optimizer.zero_grad()

            # Use the encoder and decoder directly instead of the Seq2Seq model
            # hidden = encoder(sequence_data)
            # output, hidden = decoder(sequence_data, hidden)
            
            encoded_output, decoded_output = autoencoder(sequence_data.unsqueeze(0))

            loss = criterion(decoded_output, sequence_data.unsqueeze(0))

            loss.backward()
            optimizer.step()


        # Print loss
        if step % 5000 == 0 and step > 0:
            print(f'Step {step}: Loss = {loss.item()}')
        # if step % 90000 == 0 and step > 0:
        #     print("latent vector :", encoded_output[:, -1, :])
    
        state = next_state

# Usage:

# enc = Encoder_rnn(INPUT_DIM, HID_DIM, N_LAYERS).to(device)
# dec = Decoder_rnn(OUTPUT_DIM, HID_DIM, N_LAYERS).to(device)
autoenc = RNNAutoencoder(INPUT_DIM, HID_DIM).to(device)

optimizer = optim.Adam(autoenc.parameters(), lr=learning_rate)
criterion = nn.MSELoss()
N_STEPS = 10000000

train(autoenc, env, optimizer, criterion, N_STEPS)
