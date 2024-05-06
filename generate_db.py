from metamathpy.database import parse
from metamathpy.proof import *
import os
import torch as tr
import random
from helpers import *

fpath = os.path.join("./set.mm")
db = parse(fpath)
dbdb,test,valid = [],[],[]
targs_train,targs_test,targs_valid = [],[],[]
test_labels,train_labels,valid_labels = [],[],[]
embeddings,indx_no_of_test,indx_no_of_valid = dict(),dict(),dict()

# number of statements to traverse
label_idx = 0
statements = 7000
test_cases = 150
valid_cases = 150



for i in range(test_cases):
    x = random.randrange(statements)
    while(x in indx_no_of_test):
        x = random.randrange(statements)
    indx_no_of_test[x] = True
    
for i in range(test_cases):
    x = random.randrange(statements)
    while(x in indx_no_of_test or x in indx_no_of_valid):
        x = random.randrange(statements)
    indx_no_of_valid[x] = True



# copied from proof.py, but prints the stack at every step



def extract_essentials(rule):
    essentials = []
    for x in rule.essentials:
        essentials.append(x[0])
    return essentials
    
for (i,rule) in enumerate(db.rules.values()):
    if i == statements: break
    if rule.consequent.tag != "$p" : continue
    
    essentials = extract_essentials(rule)
    
    # print(x.label)
    
    root, _ = verify_proof(db, db.rules[rule.consequent.label])
    # print(root)
    labels = extract_proof_labels(root)
    
    labels.append(rule.consequent.label)
    for label in labels:
        if label not in embeddings:
            embeddings[label] = label_idx
            label_idx += 1
    for label in essentials:
        if label not in embeddings:
            embeddings[label] = label_idx
            label_idx += 1
            
    for i,label in enumerate(essentials):
        essentials[i] = embeddings[label]
    for i,label in enumerate(labels):
        labels[i] = embeddings[label]        

    stack_labels = []
    stacks = [[]]
    
    
    for j,x in enumerate(labels):
        stack_labels.append(x)
        temp = stack_labels.copy()
        stacks.append(temp)
    # print(len(labels),labels)
    for i,stack in enumerate(stacks):
        stacks[i] = [labels[-1]] + essentials + stack
        assert len(stacks[i]) > 0
    # there is nothing to predict in the last stack
    stacks = stacks[:-1]
    # print(stacks)
    if i in indx_no_of_valid:
        valid += stacks
        valid_labels.append(rule.consequent.label)
        targs_valid += labels
    elif i in indx_no_of_test:
        test += stacks
        test_labels.append(rule.consequent.label)
        targs_test += labels
    else:    
        dbdb += stacks
        targs_train += labels
        train_labels.append(rule.consequent.label)
        # print(len(stacks),len(labels))
        
        




# from string to integer indexes in embeddings_word

#saving database and embeddings_word
# tr.save(embeddings,"embeddings.pt")

# tr.save(dbdb,"database.pt")
# tr.save(targs_train,"targs_train.pt")
# tr.save(train_labels,"train_labels.pt")

# tr.save(test,"test.pt")
# tr.save(test_labels,"test_labels.pt")
# tr.save(targs_test,"targs_test.pt")

# tr.save(valid,"valid.pt")
# tr.save(valid_labels,"valid_labels.pt")
# tr.save(targs_valid,"targs_valid.pt")


# print(f"db={len(dbdb)},labels={len(train_labels)}")
# print(f"valid={len(valid)},labels={len(valid_labels)}")
# print(f"test={len(test)},labels={len(test_labels)}")

# print(f"embeddings={len(embeddings)}")