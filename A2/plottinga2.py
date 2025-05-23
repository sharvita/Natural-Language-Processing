'''
CSCI 5832 Assignment 2
Spring 2025
The following sample code was taken from a tutorial by PyTorch and modified for our assignment.
Source: https://pytorch.org/tutorials/beginner/nlp/deep_learning_tutorial.html
'''
import torch
import random
from tqdm import tqdm
from util import *
import matplotlib.pyplot as plt

class SentimentClassifier(torch.nn.Module):

    def __init__(self, input_dim: int = 6, output_size: int = 1):
        super(SentimentClassifier, self).__init__()

        # Define the parameters that we will need.
        # Torch defines nn.Linear(), which gives the linear function z = Xw + b.
        self.linear = torch.nn.Linear(input_dim, output_size)

    def forward(self, feature_vec):
        # Pass the input through the linear layer,
        # then pass that through sigmoid to get a probability.
        z = self.linear(feature_vec)
        return torch.sigmoid(z)

model = SentimentClassifier()

# the model knows its parameters.  The first output below is X, the second is b.
# Whenever you assign a component to a class variable in the __init__ function
# of a module, which was done with the line
# self.linear = nn.Linear(...)
# Then through some Python magic from the PyTorch devs, your module
# (in this case, SentimentClassifier) will store knowledge of the nn.Linear's parameters
for param in model.parameters():
    print(param)

# To run the model, pass in a feature vector
# Here we don't need to train, so the code is wrapped in torch.no_grad()
with torch.no_grad():
    sample_feature_vector = torch.tensor([[3.0, 2.0, 1.0, 3.0, 0.0, 4.18965482711792],[3.0, 2.0, 1.0, 3.0, 0.0, 4.18965482711792]])
    log_prob = model(sample_feature_vector)
    # print('Log probability from the untrained model:', log_prob)
    # print('Label based on the log probability:', model.logprob2label(log_prob))


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# Sample training loop below. Because it uses functions that you are asked to write for the assignment,     #
# it will not run as is, and is not guaranteed to work with your existing code. You may need to modify it.  #
#                                                                                                           #
# No need to use this code if you have a better way,                                                        #
# or if you can't figure out how to make it run with your existing code.                                    #
# It is only provided here to give you an idea of how we expect you to train the model.                     #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

def logprob2label(log_prob):
    # This helper function converts the probability output of the model
    # into a binary label. Use it for the evaluation metrics.
    return log_prob.item() > 0.5

loss_function = torch.nn.BCELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
num_epochs = 100
batch_size = 16

all_texts, all_labels = load_train_data('hotelPosT-train.txt', 'hotelNegT-train.txt')
train_texts, train_labels, dev_texts, dev_labels = split_data(all_texts, all_labels)

# Featurize and normalize
#training
train_vectors = [featurize_text(text) for text in train_texts]
#train_vectors = normalize(train_vectors)

#testing
test_vectors = [featurize_text(text) for text in dev_texts]
#test_vectors = normalize(test_vectors)

train_losses = []
test_losses = []

for epoch in range(num_epochs):
    model.train()
    epoch_train_losses = []

    samples = list(zip(train_vectors, train_labels))
    random.shuffle(samples)
    batches = [samples[i:i + batch_size] for i in range(0, len(samples), batch_size)]
    for batch in tqdm(batches):
        feature_vectors, labels = zip(*batch)
        model.zero_grad()

        log_probs = model(torch.tensor(feature_vectors, dtype=torch.float32))
        labels = torch.tensor(labels, dtype=torch.float32)
        loss = loss_function(log_probs, torch.unsqueeze(labels, dim=1))
        loss.backward()
        optimizer.step()

        epoch_train_losses.append(loss.item())
    samples = list(zip(test_vectors, dev_labels))
    random.shuffle(samples)
    batches = [samples[i:i + batch_size] for i in range(0, len(samples), batch_size)]

    # Testing phase
    model.eval()
    epoch_test_losses = []

    for batch in tqdm(batches):
        feature_vectors, labels = zip(*batch)
        log_probs = model(torch.tensor(feature_vectors, dtype=torch.float32))
        labels = torch.tensor(labels, dtype=torch.float32).view(-1, 1)
        loss = loss_function(log_probs, labels)
        epoch_test_losses.append(loss.item())
    train_losses.append(sum(epoch_train_losses)/len(epoch_train_losses))
    test_losses.append(sum(epoch_test_losses)/len(epoch_test_losses))

    print('Epoch:', epoch)
    print('Avg train loss:', sum(epoch_train_losses) / len(epoch_train_losses))
    print("Test Loss: ",  sum(epoch_test_losses) / len(epoch_test_losses))
    print("Batch size:", batch_size)
    print("Learning Rate: .1")

# Plotting the training and testing loss over epochs
plt.plot(list(range(num_epochs)), train_losses, label='Training Loss')
plt.plot(list(range(num_epochs)), test_losses, label='Testing Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.title('Training and Testing Loss for each Epochs')
plt.show()
