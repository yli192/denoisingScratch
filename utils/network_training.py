import numpy as np
import torch
from torch import nn
from torch.optim import SGD


"""This module summarizes the common procedures for training a neural network model.
"""
__author__ = 'Gary Y. Li'

def network_training(
        net,
        image_train,
        label_train,
        output_image_path,
        output_model_path,
        num_epoches,
        num_input_channels = 1,
        batch_size  = 1,
        lr = 3e-5,
        momentum = 0.95
):
    image_train = torch.Tensor(np.reshape(image_train, (
        batch_size, num_input_channels, image_train.shape[0], image_train.shape[1], image_train.shape[2]))).cuda()
    label_train = torch.Tensor(np.reshape(label_train, (
    batch_size, num_input_channels, label_train.shape[0], label_train.shape[1], label_train.shape[2]))).cuda()

    optimizer = SGD(net.parameters(), lr=lr, momentum=momentum)
    optimizer.zero_grad()

    for epoch in range(num_epoches):
        # 1. make a forward pass through the network
        output = net(image_train)
        # 2. use the network output to calculate the loss
        loss = nn.L1Loss()
        l = loss(output, label_train)
        print('loss', l)
        # 3. perform a backward pass through the network with loss.backward() to calculate the gradients.
        l.backward()
        # 4. take a step with the optimizer to update the weights.
        optimizer.step()
        if epoch == num_epoches - 1:
            out_im = output.cpu().detach().numpy()
            out_im = np.squeeze(out_im, axis=0)
            out_im = np.squeeze(out_im, axis=0)
            fp = open(output_image_path + str(num_epoches) + '.npy','wb')
            np.save(fp, out_im)
            torch.save(net.state_dict(), output_model_path)

    print('Finished Training')