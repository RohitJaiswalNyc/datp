from neural_network import *
import torch as tr
from time import perf_counter
db = tr.load("database.pt")
targs_train = tr.load("targs_train.pt")
from matplotlib import pyplot as pt
from numpy import trapz
"""
Training loop
"""

def init_weights(m):
  if isinstance(m, tr.nn.Linear):
    tr.nn.init.kaiming_normal_(m.weight)
    m.bias.data.fill_(0.01)



num_updates = 200001
length = len(db)
batch_size = 64
loss_curve_train,accu_curve_train,norm_curve_train = [],[],[]


stf = StackFormer(emb_dim)
stf.apply(init_weights)
# stf.use_flash_attention=True
model_name = "./model.pt"
# stf.load_state_dict(tr.load(model_name))

loss_fn = tr.nn.CrossEntropyLoss()
opt = tr.optim.Adam(stf.parameters(), lr=0.0001)


print(length)

start = perf_counter()
def train():
  index = 0
  
  for update in range(num_updates):

    # prepare training batch
    stacks, goals, targs = [], [], []
    for b in range(batch_size):
      stacks_b = db[index]
      targs.append(targs_train[index])
      index += 1
      index = index % length
      goal_b = stacks_b[0] # inputs are stacks up to last step
      goals_b = [[goal_b]]

      stacks.append(stacks_b)
      goals += goals_b
      
      
    # forward
    goals, targs = tr.tensor(goals), tr.tensor(targs)
    logits = stf(stacks, goals)
    
    loss = loss_fn(logits, targs)
    loss_curve_train.append(loss.item())
    accu_curve_train.append((logits.argmax(dim=-1) == targs).to(float).mean().item())

    # backward
    opt.zero_grad()
    loss.backward()
    opt.step()

    total_norm = 0
    parameters = [p for p in stf.parameters() if p.grad is not None and p.requires_grad]
    for p in parameters:
        param_norm = p.grad.detach().data.norm(2)
        total_norm += param_norm.item() ** 2
    total_norm = total_norm ** 0.5
    norm_curve_train.append(total_norm)
    
    # progress
    if update % 100 == 0: 
      print(f"update {update}: loss={loss_curve_train[-1]}, accu={accu_curve_train[-1]}, lr={norm_curve_train[-1]}")
      tr.cuda.empty_cache()
    
    # do if you are not sure if your program will crash due to out of memory and stuff
    # if update % 1000 == 0:
    #   tr.save(stf.state_dict(), model_name)
    
  print(f"total time = {perf_counter()-start}s")    
    
train()
tr.save(stf.state_dict(), model_name)  
print(accu_curve_train.mean(),loss_curve_train.mean())

fig = pt.figure(figsize=(8,3))
pt.subplot(1,3,1)
pt.plot(loss_curve_train)
pt.ylabel("cross entropy loss")
pt.subplot(1,3,2)
pt.plot(accu_curve_train)
pt.ylabel("accuracy on batch")
pt.subplot(1,3,3)
pt.plot(norm_curve_train)
pt.ylabel("learning rate on batch")
fig.supxlabel("update")
_ = pt.tight_layout()
pt.show()


#heads = 32
# accu_area = 366.4453125
# loss_area = 1979.934962096624
# total time = 27.408820499999997s
#heads = 16
# accu_area = 375.3046875
# loss_area = 1984.307310411954
# total time = 27.114697s
#heads = 8
# accu_area = 367.21875
# loss_area = 2135.2286036359146
# total time = 26.749032299999996s


# update 0: loss=117.15303802490234, accu=0.09375, lr=405.4313842625338
# update 100: loss=4.932820796966553, accu=0.53125, lr=65.49128213966591
# update 200: loss=3.176994562149048, accu=0.71875, lr=61.1995525177864
# update 300: loss=1.0958753824234009, accu=0.84375, lr=52.61735451629449
# update 400: loss=0.7626879215240479, accu=0.90625, lr=38.53369893813492
# update 500: loss=0.3974568843841553, accu=0.96875, lr=26.550581695827216

# update 0: loss=96.02867889404297, accu=0.0625, lr=373.89755185584664
# update 100: loss=5.621600151062012, accu=0.484375, lr=66.8943255950965
# update 200: loss=2.1301770210266113, accu=0.78125, lr=67.43676410282161
# update 300: loss=0.7378603219985962, accu=0.921875, lr=38.3479782652726
# update 400: loss=0.5008405447006226, accu=0.90625, lr=39.19711368561896
# update 500: loss=0.51081383228302, accu=0.921875, lr=37.51085681004972

# update 0: loss=120.96932983398438, accu=0.09375, lr=387.78475212327214
# update 100: loss=28.342041015625, accu=0.203125, lr=77.7415957404829
# update 200: loss=22.322858810424805, accu=0.09375, lr=100.06774799799501
# update 300: loss=21.339359283447266, accu=0.078125, lr=123.27933176590903
# update 400: loss=17.29929542541504, accu=0.15625, lr=61.10563187788141
# update 500: loss=7.817487716674805, accu=0.5, lr=93.61080058861378
# update 600: loss=4.290363311767578, accu=0.46875, lr=50.92844321630659
# update 700: loss=7.695230007171631, accu=0.34375, lr=35.388730591056174
# update 800: loss=5.547111511230469, accu=0.1875, lr=74.29776224008468
# update 900: loss=9.395602226257324, accu=0.171875, lr=85.91133461794509
# update 1000: loss=8.362957954406738, accu=0.28125, lr=53.3323338338573
# update 1100: loss=19.824649810791016, accu=0.1875, lr=48.52929581156264
# update 1200: loss=13.930058479309082, accu=0.21875, lr=62.59073239179192
# update 1300: loss=15.002283096313477, accu=0.046875, lr=51.8429795182757
# update 1400: loss=11.615994453430176, accu=0.046875, lr=61.12447084683684
# update 1500: loss=17.229778289794922, accu=0.0625, lr=48.79226812583361
# update 1600: loss=11.625104904174805, accu=0.1875, lr=27.91689381826552
# update 1700: loss=13.393369674682617, accu=0.109375, lr=39.73106998560917
# update 1800: loss=14.22840404510498, accu=0.078125, lr=44.53251796890313
# update 1900: loss=17.129898071289062, accu=0.0625, lr=39.489779804586746
# update 2000: loss=10.609845161437988, accu=0.125, lr=49.971686936080786
# update 2100: loss=17.96796989440918, accu=0.078125, lr=67.26227831661654
# update 2200: loss=17.70960235595703, accu=0.078125, lr=33.11215807139747
# update 2300: loss=21.543453216552734, accu=0.078125, lr=33.54812409205153
# update 2400: loss=16.02998924255371, accu=0.15625, lr=31.249432703212637
# update 2500: loss=9.245091438293457, accu=0.09375, lr=32.392503050712165
# update 2600: loss=21.74860382080078, accu=0.015625, lr=29.124226911512682
# update 2700: loss=10.49288558959961, accu=0.0625, lr=24.644314611054046
# update 2800: loss=17.153789520263672, accu=0.03125, lr=31.61570298919606
# update 2900: loss=12.775247573852539, accu=0.125, lr=30.651379570294772
# update 3000: loss=11.925397872924805, accu=0.0625, lr=35.98298598225957
# update 3100: loss=12.4496431350708, accu=0.0625, lr=49.0925657095849
# update 3200: loss=23.788719177246094, accu=0.0625, lr=35.63569962863205
# update 3300: loss=17.264751434326172, accu=0.21875, lr=45.18962206917348
# update 3400: loss=11.441675186157227, accu=0.125, lr=39.23403242893014
# update 3500: loss=15.930720329284668, accu=0.15625, lr=38.721412326899504
# update 3600: loss=9.80988883972168, accu=0.09375, lr=41.89124507530433
# update 3700: loss=3.5678329467773438, accu=0.3125, lr=31.288255503100334
# update 3800: loss=3.2227959632873535, accu=0.4375, lr=22.265148312575704
# update 3900: loss=3.9180712699890137, accu=0.40625, lr=29.57299574296319
# update 4000: loss=3.1329197883605957, accu=0.328125, lr=29.13806937936081
# update 4100: loss=13.595552444458008, accu=0.0625, lr=81.57404283182314
# update 4200: loss=3.3210434913635254, accu=0.1875, lr=36.30395792089369