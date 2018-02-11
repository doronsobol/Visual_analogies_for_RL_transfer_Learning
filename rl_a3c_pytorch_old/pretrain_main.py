from __future__ import print_function, division
import os
os.environ["OMP_NUM_THREADS"] = "1"
import argparse
import sys
import torch
import  torch.optim as optim
from torch.multiprocessing import Process, Lock
from model import static_A3Clstm
import time
from dataset_image import dataset_image
import torch.nn.functional as F
from torch.autograd import Variable
import tensorboard
from tensorboard import summary
from tqdm import tqdm, trange
from pdb import set_trace as st
from logger import Logger
import numpy as np
from sklearn.metrics import accuracy_score

parser = argparse.ArgumentParser(description='A3C')
parser.add_argument(
    '--epochs', 
    default=100, 
    type=int,
    metavar='LG', 
    help='The batch size')
parser.add_argument(
    '--gradient-clip', 
    default=10, 
    type=float,
    metavar='LG', 
    help='The batch size')
parser.add_argument(
    '--batch-size', 
    default=16, 
    type=int,
    metavar='LG', 
    help='The batch size')
parser.add_argument(
    '--sec-len', 
    default=5, 
    type=int,
    metavar='LG', 
    help='The batch size')
parser.add_argument(
    '--num_of_targets', 
    default=6, 
    type=int,
    metavar='LG', 
    help='The batch size')
parser.add_argument(
    '--experiment_name', 
    default='name', 
    metavar='LG', 
    help='The batch size')
parser.add_argument(
    '--dataset-file', 
    default='demon_labels.txt.bak', 
    metavar='LG', 
    help='The batch size')
parser.add_argument(
    '--log_dir', 
    default='log_dir', 
    metavar='LG', 
    help='The batch size')
parser.add_argument(
    '--optimizer', 
    default='Adam', 
    metavar='LG', 
    help='The batch size')
parser.add_argument(
    '--save_dir', 
    default='pretrained_models', 
    metavar='LG', 
    help='The batch size')
parser.add_argument(
    '--lr', 
    default=0.0001, 
    type=float,
    metavar='LG', 
    help='The batch size')
parser.add_argument(
    '--root', 
    default='./', 
    metavar='LG', 
    help='The batch size')
parser.add_argument(
    '--loss-type', 
    default='cross_entropy', 
    metavar='LG', 
    help='The batch size')
parser.add_argument(
    '--translate', 
    action='store_true',
    help='The batch size')
parser.add_argument(
    '--cuda', 
    default=True, 
    metavar='LG', 
    help='The batch size')


# Based on
# https://github.com/pytorch/examples/tree/master/mnist_hogwild
# Training settings
# Implemented multiprocessing using locks but was not beneficial. Hogwild
# training was far superior
def test(model, dataloader, writer, criterion):
    global args
    epoch_data = tqdm(enumerate(dataloader), leave=False)
    all_losses = None
    all_accuracies = None
    for it, (images, labels) in epoch_data:

        images = Variable(images)
        labels = Variable(labels)

        if args.cuda:
            images = images.cuda()
            labels = labels.cuda()
        labels = labels.view(labels.size(0)*labels.size(1), -1)
        labels = labels.squeeze()

        logits, features = model(images)
        if 'l1' ==  args.loss_type or 'l2' ==  args.loss_type:
            logits = features

        loss = criterion(logits, labels)

        #labels = to_np(labels)
        if labels.dim() > 1:
            labels = to_np(torch.max(labels, 1)[1])
        else:
            labels = to_np(labels)

        loss = to_np(loss)

        prediction = to_np(torch.max(logits, 1)[1])
        accuracy = np.array([accuracy_score(labels, prediction)])
        if all_losses is None:
            all_losses = loss
            all_accuracies = accuracy
        else:
            all_accuracies = np.concatenate((all_accuracies, accuracy))
            all_losses = np.concatenate((all_losses, loss))
        if 0 == it % 10:
            writer.set_description("Test Loss: {:.6f}, Test Accuracy: {:.6f}".format(
                                    loss[0], accuracy[0]))
    
    return np.mean(all_losses), np.mean(all_accuracies)


def to_np(x):
    return x.data.cpu().numpy()

def soft_cross_entropy(logits, labels):
    log_softmax = F.log_softmax(logits)
    cross_entropy = torch.sum(-(labels * log_softmax), dim=1)
    return torch.mean(cross_entropy)

if __name__ == '__main__':
    args = parser.parse_args()
    torch.set_default_tensor_type('torch.FloatTensor')

    model = static_A3Clstm(1, args.num_of_targets)
    if args.cuda:
        model = model.cuda()
    print(model)
    print(args)
    if args.optimizer == 'RMSprop':
        optimizer = optim.RMSprop(model.parameters(), lr=args.lr)
    elif args.optimizer == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
    else:
        raise NotImplementedError("Unknown Optimizer: {}".format(args.optimizer))
    
    logger = Logger("{}/{}".format(args.log_dir, args.experiment_name))

    if 'cross_entropy' == args.loss_type:
        criterion = torch.nn.CrossEntropyLoss()
        is_hard = True
    elif 'KL_div' ==  args.loss_type:
        is_hard = False
        criterion = torch.nn.KLDivLoss()
    elif 'l1' ==  args.loss_type:
        is_hard = False
        criterion = torch.nn.L1Loss()
    elif 'l2' ==  args.loss_type:
        is_hard = False
        criterion = torch.nn.MSELoss()
    elif 'soft_cross_entropy' ==  args.loss_type:
        is_hard = False
        criterion = soft_cross_entropy
    else:
        raise NotImplementedError("Unknown loss: {}".format(args.loss_type))
    
    dataset = dataset_image(dataset_file=args.dataset_file, is_max=is_hard, is_test=False, sec_len=args.sec_len, use_translation=args.translate, root=args.root) 
    dataloader = torch.utils.data.DataLoader(dataset=dataset, batch_size=args.batch_size, shuffle=True, num_workers=8)
    dataset_test = dataset_image(dataset_file=args.dataset_file, is_max=is_hard, is_test=True, sec_len=args.sec_len, use_translation=args.translate, root=args.root) 
    dataloader_test = torch.utils.data.DataLoader(dataset=dataset, batch_size=args.batch_size, num_workers=2)
    all_labels = [] 
    #for im, l in tqdm(dataloader):
    #    if all_labels == []:
    #        all_labels = l.numpy()
    #    else:
    #        all_labels = np.concatenate((all_labels, l.numpy()))

    #element_counter = np.unique(all_labels, return_counts=True)[1]
    #pdb.set_trace()
    #print(element_counter/np.sum(element_counter))
    #sys.exit(0)

    iterations = 0
    epochs = trange(args.epochs)
    for ep in epochs:
        epoch_data = tqdm(dataloader, leave=False)
        for images, labels in epoch_data:

            images = Variable(images)
            labels = Variable(labels)

            if args.cuda:
                images = images.cuda()
                labels = labels.cuda()
            #labels = labels.view(-1)
            labels = labels.view(labels.size(0)*labels.size(1), -1)
            labels = labels.squeeze()

            logits, features = model(images)
            if 'l1' ==  args.loss_type or 'l2' ==  args.loss_type:
                logits = features
            #log_softmax = F.log_softmax(logits)
            #loss = KL_loss(log_softmax, labels)

            loss = criterion(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm(model.parameters(), args.gradient_clip)
            optimizer.step()
            iterations += 1

            if 0 == iterations%10:
                logger.scalar_summary('loss', to_np(loss), iterations)
                epochs.set_description("Loss: {:.6f}".format(loss.cpu().data.numpy()[0]))

                for tag, value in model.named_parameters():
                    if not 'critic' in tag and not 'actor' in tag:
                        tag = tag.replace('.', '/')
                        logger.histo_summary(tag, to_np(value), iterations)
                        logger.histo_summary(tag+'/grad', to_np(value.grad), iterations)
            if 0 == iterations%1000:
                save_file = os.path.join(args.save_dir, 'checkpoint-{}-{}.dat'.format(args.experiment_name ,iterations))
                state_to_save = model.state_dict()
                torch.save(state_to_save, save_file)
                tqdm.write("Saving checkpoint: {:.6f}".format(loss.cpu().data.numpy()[0]))
                #print('Labels: {}, Logits: {}'.format(labels, F.softmax(logits)) )
                #sys.exit(0)
        test_loss, test_accuracy = test(model, dataloader_test, epochs, criterion)

        epochs.write("Test Loss: {:.6f}".format(test_loss))
        epochs.write("Test Accuracy: {:.6f}".format(test_accuracy))
        logger.scalar_summary('test_loss', test_loss, ep + 1)
        logger.scalar_summary('test_accuracy', test_accuracy, ep + 1)
