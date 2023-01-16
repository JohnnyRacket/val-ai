import torch
import random
import numpy as np
import NeuralNetwork
import GameState
from filter_images import resize_and_grayscale_img

print(torch.cuda.is_available())

print(torch.cuda.device_count())

print(torch.cuda.current_device())

print(torch.cuda.device(0))

print(torch.cuda.get_device_name(0))

model = NeuralNetwork()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-6)
criterion = torch.nn.MSELoss()

game_state = GameState()
replay_memory = []

action = torch.zeros([model.number_of_actions], dtype=torch.float32)
action[1] = 1
img, reward, terminal = game_state.frame_step(action)
img = resize_and_grayscale_img(img)
transform = torch.transforms.ToTensor()
tensor = transform(img)
state = torch.cat((tensor, tensor, tensor, tensor)).unsqueeze(0)

epsilon = model.initial_epsilon
iteration = 0

while iteration < model.number_of_iterations:
    # get output from the neural network
    output = model(state)[0]

    # initialize action
    action = torch.zeros([model.number_of_actions], dtype=torch.float32)
    if torch.cuda.is_available():  # put on GPU if CUDA is available
        action = action.cuda()

    # epsilon greedy exploration
    random_action = random.random() <= epsilon
    if random_action:
        print("Performed random action!")
    action_index = [torch.randint(model.number_of_actions, torch.Size([]), dtype=torch.int)
                    if random_action
                    else torch.argmax(output)][0]

    if torch.cuda.is_available():  # put on GPU if CUDA is available
        action_index = action_index.cuda()

    action[action_index] = 1

    # get next state and reward
    img1, reward, terminal = game_state.frame_step(action)
    img1 = resize_and_grayscale_img(img1)
    transform = torch.transforms.ToTensor()
    tensor = transform(img1)
    state_1 = torch.cat((state.squeeze(0)[1:, :, :], img1)).unsqueeze(0)

    action = action.unsqueeze(0)
    reward = torch.from_numpy(np.array([reward], dtype=np.float32)).unsqueeze(0)

    # save transition to replay memory
    replay_memory.append((state, action, reward, state_1, terminal))

    # if replay memory is full, remove the oldest transition
    if len(replay_memory) > model.replay_memory_size:
        replay_memory.pop(0)

    # epsilon annealing
    epsilon = epsilon_decrements[iteration]

    # sample random minibatch
    minibatch = random.sample(replay_memory, min(len(replay_memory), model.minibatch_size))

    # unpack minibatch
    state_batch = torch.cat(tuple(d[0] for d in minibatch))
    action_batch = torch.cat(tuple(d[1] for d in minibatch))
    reward_batch = torch.cat(tuple(d[2] for d in minibatch))
    state_1_batch = torch.cat(tuple(d[3] for d in minibatch))

    if torch.cuda.is_available():  # put on GPU if CUDA is available
        state_batch = state_batch.cuda()
        action_batch = action_batch.cuda()
        reward_batch = reward_batch.cuda()
        state_1_batch = state_1_batch.cuda()

    # get output for the next state
    output_1_batch = model(state_1_batch)

    # set y_j to r_j for terminal state, otherwise to r_j + gamma*max(Q)
    y_batch = torch.cat(tuple(reward_batch[i] if minibatch[i][4]
                              else reward_batch[i] + model.gamma * torch.max(output_1_batch[i])
                              for i in range(len(minibatch))))

    # extract Q-value
    q_value = torch.sum(model(state_batch) * action_batch, dim=1)

    # PyTorch accumulates gradients by default, so they need to be reset in each pass
    optimizer.zero_grad()

    # returns a new Tensor, detached from the current graph, the result will never require gradient
    y_batch = y_batch.detach()

    # calculate loss
    loss = criterion(q_value, y_batch)

    # do backward pass
    loss.backward()
    optimizer.step()

    # set state to be state_1
    state = state_1