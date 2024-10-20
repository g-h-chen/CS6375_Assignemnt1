import numpy as np
import torch
import torch.nn as nn
from torch.nn import init
import torch.optim as optim
import math
import random
import os
import time
from tqdm import tqdm
import json
import string
from argparse import ArgumentParser
import pickle
import pdb

unk = '<UNK>'
# Consult the PyTorch documentation for information on the functions used below:
# https://pytorch.org/docs/stable/torch.html
class RNN(nn.Module):
    def __init__(self, input_dim, h, n_layers):  # Add relevant parameters
        super(RNN, self).__init__()
        self.h = h
        self.numOfLayer = n_layers
        self.rnn = nn.RNN(input_dim, h, self.numOfLayer, nonlinearity='tanh')
        self.W = nn.Linear(h, 5) # output
        self.softmax = nn.LogSoftmax(dim=-1) # should be -1, not 1
        self.loss = nn.NLLLoss()

    def compute_Loss(self, predicted_vector, gold_label):
        return self.loss(predicted_vector, gold_label)

    def forward(self, inputs):
        '''
        inputs: (seqlen, bsz=1, dim=50)
        '''
        # [to fill] obtain hidden layer representation (https://pytorch.org/docs/stable/generated/torch.nn.RNN.html)
        output, hidden = self.rnn(inputs) 
        # output: (seqlen, bsz, hidden_dim)
        # hidden: (n_layer, bsz=1, hidden_dim)
        # output[-1] == hidden[-1]

        # [to fill] obtain output layer representations
        # out = self.W(output) # (seqlen, bsz=1, 5)
        out = self.W(output) # (seqlen, bsz=1, 5)

        # [to fill] sum over output 
        out = out.sum(0).squeeze(0) # (5,)

        # [to fill] obtain probability dist.
        out = self.softmax(out)
        return out


        '''
        # [to fill] obtain hidden layer representation (https://pytorch.org/docs/stable/generated/torch.nn.RNN.html)
        # seqlen, bsz, input_dim = inputs.size() # note that input_dim can be different from hidden dim
        # h0 = torch.zeros(self.numOfLayer, bsz, self.h)

        _, hidden = self.rnn(inputs) 
        # _: (seqlen, bsz, hidden_dim)
        # hidden: (n_layer, bsz=1, hidden_dim)
        
        # [to fill] obtain output layer representations
        out = self.W(hidden[-1].squeeze(0)) # (hidden_dim,) -> (5,)

        # [to fill] obtain probability dist.
        out = self.softmax(out) # (5,)
        '''



def load_data(train_data, val_data):
    with open(train_data) as training_f:
        training = json.load(training_f)
    with open(val_data) as valid_f:
        validation = json.load(valid_f)

    tra = []
    val = []
    for elt in training:
        tra.append((elt["text"].split(),int(elt["stars"]-1)))
    for elt in validation:
        val.append((elt["text"].split(),int(elt["stars"]-1))) # tuple(str: text, int: label)
    return tra, val


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-hd", "--hidden_dim", type=int, required = True, help = "hidden_dim")
    parser.add_argument("--n_layers", type=int, required = True, help = "n_layers")


    parser.add_argument("-e", "--epochs", type=int, required = True, help = "num of epochs to train")
    parser.add_argument("--train_data", required = True, help = "path to training data")
    parser.add_argument("--val_data", required = True, help = "path to validation data")
    parser.add_argument("--test_data", default = "to fill", help = "path to test data")
    parser.add_argument('--do_train', action='store_true')
    args = parser.parse_args()

    device = 'cuda:0'

    print("========== Loading data ==========")
    train_data, valid_data = load_data(args.train_data, args.val_data) # X_data is a list of pairs (document, y); y in {0,1,2,3,4}

    # # for debugging
    # train_data = random.sample(train_data, 48)
    # valid_data = random.sample(valid_data, 64)

    # Think about the type of function that an RNN describes. To apply it, you will need to convert the text data into vector representations.
    # Further, think about where the vectors will come from. There are 3 reasonable choices:
    # 1) Randomly assign the input to vectors and learn better embeddings during training; see the PyTorch documentation for guidance
    # 2) Assign the input to vectors using pretrained word embeddings. We recommend any of {Word2Vec, GloVe, FastText}. Then, you do not train/update these embeddings.
    # 3) You do the same as 2) but you train (this is called fine-tuning) the pretrained embeddings further.
    # Option 3 will be the most time consuming, so we do not recommend starting with this

    print("========== Vectorizing data ==========")
    model = RNN(50, args.hidden_dim, n_layers=args.n_layers).to(device)  # Fill in parameters
    # optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    # optimizer = optim.Adam(model.parameters(), lr=1e-3)
    word_embedding = pickle.load(open('./word_embedding.pkl', 'rb'))

    stopping_condition = False
    epoch = 0

    last_train_accuracy = 0
    last_validation_accuracy = 0
    epoch_losses_train = []
    epoch_losses_val = []

    epoch_accs_train = []
    epoch_accs_val = []

    # while not stopping_condition:
    for _ in range((args.epochs)):
        random.shuffle(train_data)
        model.train()
        # You will need further code to operationalize training, ffnn.py may be helpful
        print("Training started for epoch {}".format(epoch + 1))
        train_data = train_data
        correct = 0
        total = 0
        minibatch_size = 16
        N = len(train_data)

        loss_total = 0
        loss_count = 0

        pbar = tqdm(range(N // minibatch_size))
        epoch_loss_train = 0
        for minibatch_index in pbar:
            optimizer.zero_grad()
            loss = None
            for example_index in range(minibatch_size):
                input_words, gold_label = train_data[minibatch_index * minibatch_size + example_index]
                input_words = " ".join(input_words)

                # Remove punctuation
                input_words = input_words.translate(input_words.maketrans("", "", string.punctuation)).split()

                # Look up word embedding dictionary
                vectors = [word_embedding[i.lower()] if i.lower() in word_embedding.keys() else word_embedding['unk'] for i in input_words ]

                # Transform the input into required shape
                vectors = torch.tensor(vectors).view(len(vectors), 1, -1)
                
                output = model(vectors.to(device))

                # Get loss
                example_loss = model.compute_Loss(output.view(1,-1), torch.tensor([gold_label]).to(device))
                # example_loss = nn.CrossEntropyLoss()(output.view(1,-1), torch.tensor([gold_label]).to(device))

                # Get predicted label
                predicted_label = torch.argmax(output)

                correct += int(predicted_label == gold_label)
                # print(predicted_label, gold_label)
                # print(input_words)
                # print('-'*10)
                total += 1
                if loss is None:
                    loss = example_loss
                else:
                    loss += example_loss

            # pdb.set_trace()
            loss = loss / minibatch_size
            loss_total += loss.data
            loss_count += 1
            loss.backward()
            optimizer.step()

            # pbar.update(1)
            pbar.postfix = f'overall acc: {correct/total:.4f}, overall loss: {loss_total/loss_count:.4f}'
            # pdb.set_trace()

        epoch_losses_train.append(loss_total.item()/loss_count)
        epoch_accs_train.append(correct/total)

        print(loss_total/loss_count)
        print("Training completed for epoch {}".format(epoch + 1))
        print("Training accuracy for epoch {}: {}".format(epoch + 1, correct / total))
        trainning_accuracy = correct/total


        model.eval()
        correct = 0
        total = 0
        random.shuffle(valid_data)
        print("Validation started for epoch {}".format(epoch + 1))
        valid_data = valid_data

        epoch_loss_val = 0
        for input_words, gold_label in tqdm(valid_data):
            input_words = " ".join(input_words)
            input_words = input_words.translate(input_words.maketrans("", "", string.punctuation)).split()
            vectors = [word_embedding[i.lower()] if i.lower() in word_embedding.keys() else word_embedding['unk'] for i
                       in input_words]

            vectors = torch.tensor(vectors).view(len(vectors), 1, -1)
            output = model(vectors.to(device))

            example_loss = model.compute_Loss(output.view(1,-1), torch.tensor([gold_label]).to(device))
            epoch_loss_val += example_loss.item()

            predicted_label = torch.argmax(output)
            correct += int(predicted_label == gold_label)
            total += 1
            # print(predicted_label, gold_label)
        epoch_losses_val.append(epoch_loss_val/len(valid_data))
        epoch_accs_val.append(correct/total)

        print("Validation completed for epoch {}".format(epoch + 1))
        print("Validation accuracy for epoch {}: {}".format(epoch + 1, correct / total))
        validation_accuracy = correct/total

        # if validation_accuracy < last_validation_accuracy and trainning_accuracy > last_train_accuracy:
        #     # stopping_condition=True
        #     print("Training done to avoid overfitting!")
        #     print("Best validation accuracy is:", last_validation_accuracy)
        # else:
        #     last_validation_accuracy = validation_accuracy
        #     last_train_accuracy = trainning_accuracy

        epoch += 1

    
    output_pth = f'outputs/rnn/dim{args.hidden_dim}_L{args.n_layers}.json'

    with open(output_pth, 'w') as f:
        data = {
            'epoch_losses_train':epoch_losses_train,
            'epoch_losses_val':epoch_losses_val,

            'epoch_accs_train': epoch_accs_train,
            'epoch_accs_val': epoch_accs_val,
        }
        json.dump(data, f, indent=1, ensure_ascii=False)




    # You may find it beneficial to keep track of training accuracy or training loss;

    # Think about how to update the model and what this entails. Consider ffnn.py and the PyTorch documentation for guidance
