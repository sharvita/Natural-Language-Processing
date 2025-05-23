'''
CSCI 5832 Assignment 3
Spring 2025
Use this code as a starting point or reference for the neural network design
and training portion of your assignment.
'''

# FF net with an nn.Embedding layer

import torch
import torch.nn as nn
import torch.optim as optim
from a3datascaffolding import *
from a3explorescaffolding import *

class Net(nn.Module):
    def __init__(self, input_size: int, output_size: int, hidden_size: int, num_embeddings: int, embedding_dim: int, 
                 padding_idx: int = 0, pretrained_embeddings: torch.Tensor = None, freeze_embeddings: bool = False):
        super(Net, self).__init__()
        if pretrained_embeddings is not None:
            self.embedding = nn.Embedding.from_pretrained(pretrained_embeddings, padding_idx=padding_idx, freeze=freeze_embeddings)
        else:
            self.embedding = nn.Embedding(num_embeddings=num_embeddings, embedding_dim=embedding_dim, padding_idx=padding_idx)
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.activation = torch.nn.ReLU()
        self.linear2 = torch.nn.Linear(hidden_size, output_size)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        x = self.embedding(x)
        x = torch.mean(x, dim=1) # take the mean of the embeddings
        # finish the code for the forward pass
        x = self.activation(self.linear1(x))
        x = self.linear2(x)
        return self.sigmoid(x)

class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float('inf')

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False
    
#################### Training Loop ####################

# Hyperparameters that worked for TA on a CPU:
d = 50
h = 50
batch_size = 16
num_epochs = 5
learning_rate = 0.001

# Initialize the model
model = Net(
    input_size=d,
    output_size=1,
    hidden_size=h,
    num_embeddings=len(token2index),
    embedding_dim=d,
    padding_idx=0
)

# Define the loss function and the optimizer
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
early_stopper = EarlyStopper(3,0.01)

train_loader, val_loader, test_loader = getLoaders()

# Train the model
for epoch in range(num_epochs):

    for i, batch in enumerate(train_loader):
        model.train()
        # your code here: call the forward pass,
        # compute the loss, and update the model
        inputs, targets = batch
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs.squeeze(), targets.float().squeeze())
        loss.backward()
        optimizer.step()


    # Validation
    model.eval()
    val_losses = []
    # Extract the input and target variables from each validation batch
    for input_, target in val_loader:
        with torch.no_grad():
            # your code here: call the forward pass and compute
            # the loss on the validation set. Add it to val_losses
            # to compute the average loss for the entire validation set
            outputs = model(input_)
            loss = criterion(outputs.squeeze(), target.float().squeeze())
            val_losses.append(loss.item())

    val_loss = sum(val_losses) / len(val_losses)
    if early_stopper.early_stop(val_loss):
        break

    print('training loss: %.4f' % loss.item()) # Please print the training and validation loss to ensure 
                                               # you're meeting the criteria specified in the writeup
    print('validation loss: %.4f' % val_loss)


#################### Evaluation ####################

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Evaluate the model
for batch in test_loader:
    test_inputs, test_targets = batch
with torch.no_grad():
    test_outputs = model(test_inputs)

predictions = test_outputs.view(-1)
predictions = torch.tensor([1 if x >= 0.5 else 0 for x in predictions])
print(f'accuracy: {accuracy_score(test_targets, predictions)}')
print(f'precision: {precision_score(test_targets, predictions)}')
print(f'recall: {recall_score(test_targets, predictions)}')
print(f'f1: {f1_score(test_targets, predictions)}')


# ################## Getting GloVe embeddings ##################
#
# import gensim.downloader as api
# embedding_dim = d
# word_vectors = api.load(f"glove-wiki-gigaword-{embedding_dim}")
# token2index, index2token = getConversionDict()
# # Create the embedding matrix with the pretrained embeddings and the token2index dictionary
# embedding_matrix = torch.zeros(len(token2index), embedding_dim)
# for token, index in token2index.items():
#     if token in word_vectors:
#         embedding_matrix[index] = torch.tensor(word_vectors[token])
#
#
# gloveModel =   Net(
#         input_size=d,
#         output_size=1,
#         hidden_size=h,
#         num_embeddings=len(dataProcessor.token2index),
#         embedding_dim=d,
#         padding_idx=0,
#         pretrained_embeddings= embedding_matrix,
#         freeze_embeddings= True
#     )



print(k_nearest_neighbors(model.embedding.weight.data,token2index,'performance'))

plot_embeddings_3d(model.embedding.weight.data,token2index,index2token)

plot_embeddings_tsne(model.embedding.weight.data, token2index, index2token)