import torch as tr
from metamathpy.proof import *
device = tr.device("cuda:0" if tr.cuda.is_available() else "cpu")
tr.set_default_device(device)
from helpers import *
from neural_network import StackFormer,emb_dim
from metamathpy.environment import Environment
from metamathpy.database import parse
import os


fpath = os.path.join("./set.mm")
db = parse(fpath)
embeddings = tr.load("./embeddings.pt")
env = Environment(db)
idx2label = list(embeddings)
test_labels = tr.load("./test_labels.pt")

model_name = "./model.pt"
stf = StackFormer(emb_dim)
stf.load_state_dict(tr.load(model_name))
stf.eval()

print(len(test_labels))
def beam_search(stf, root_env, beam_size, max_depth,prefix):
  solution = None # successful proof if it has been found
  
  # initial environment for beam search
  beam = [root_env]
  log_probs = tr.zeros(len(beam))

  for depth in range(max_depth):

    # form current partial proofs into batch
    proofs = tr.tensor([prefix.copy() + [embeddings[tok] for tok in env.proof] for env in beam])
    goals = tr.tensor([[embeddings[env.claim.consequent.label]] for env in beam])

    # get stf predictions on entire beam as batch
    logits = stf(proofs, goals)
    # get log probabilities for proofs after each choice, broadcasts
    log_probs = log_probs[:,None] + tr.nn.functional.log_softmax(logits, dim=-1)
    
    # sort all predictions across beam from best to worst
    sort_idx = tr.argsort(log_probs.flatten(), descending=True)

    # convert flat index back to batch index and prediction
    beam_idx, prediction = tr.unravel_index(sort_idx, log_probs.shape)

    # populate new beam from best to worst
    new_beam = []
    new_log_probs = []
    
    for (b, pred) in zip(beam_idx, prediction):
      pred = int(pred)
      rule_label = idx2label[pred]
      
      # stop if beam is full
      if len(new_beam) == beam_size: break

      # try applying rule
      env = beam[b].copy()
      
      
      (_, proof, stack), msg = env.step(rule_label)

      
      # skip invalid rules and empty stacks
      if msg != "": continue
      if len(stack) == 0: continue

      # if goal was predicted and matches stack, search is done
      if stack[-1].conclusion == tuple(env.claim.consequent.tokens): 
        solution = proof
        break

      # add to beam
      new_beam.append(env)
      new_log_probs.append(log_probs[b,pred])

    # overwrite previous iteration
    beam, log_probs = new_beam, tr.tensor(new_log_probs)

    # stop early if result has been proved
    if solution is not None: break

    # stop early if beam is empty (no valid rules predicted)
    if len(beam) == 0: break

  if solution == None:
    # solution not found
    return 0
  else:
    # check result
    env = root_env.copy()
    for d, label in enumerate(solution):

      # apply rule
      _, msg = env.step(label)
      if msg != "":
        return 0

    # make sure proof succeeded
    assert stack[-1].conclusion == tuple(env.claim.consequent.tokens)
    return 1



# claim = db.rules["id"]
# proof_root, _ = verify_proof(db, claim)

# labels = extract_proof_labels(proof_root)



# env.reset(claim)
# env.step(labels[0])
# print(beam_search(stf, env, beam_size=10, max_depth=len(labels)))


accu = 0

for idx,x in enumerate(test_labels):
    # print(idx2label[x[-1][0]])
    claim = db.rules[x]
    
    proof_root = [x]
    root, _ = verify_proof(db, db.rules[x])
    labels = extract_proof_labels(root)
    essentials = [x.label for x in claim.essentials]
    essentials = [claim.consequent.label] + essentials
    
    # essentials = [embeddings[x] for x in essentials]
    # env.reset(claim)
    # env.step(labels[0])
    # ret = beam_search(stf, env, beam_size=10, max_depth=len(labels),prefix=essentials)
    # accu += ret
    print(f"proof={idx},length_of_proof= {len(labels)},proved={0},proof={x},essentials={len(essentials)}")
    
print(accu/10,accu)
