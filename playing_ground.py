# import torch as tr
from neural_network import *
import torch as tr
# import os
# from metamathpy.database import parse

# embeddings = tr.load("embeddings.pt")
db = tr.load("database.pt")
test = tr.load("test.pt")
targs = tr.load("targs_train.pt")
train_labels = tr.load("train_labels.pt")
valid = tr.load("valid.pt")
valid_labels = tr.load("valid_labels.pt")
test_labels = tr.load("test_labels.pt")

# print(len(test),len(targs),len(valid),len(train_labels),len(valid_labels),len(test_labels))

# tokens as input
# model_parameters = 9300480
# len(test.pt) = 23927
# len(db.pt) = 437420
# len(test_labels) = 284
# len(train_labels) = 5325
# max_sequence_lenght = 7224
# maximum hypothesis dependency for a proof = 71
# total proofs in set.mm = 101831