# SGDVT: Social Generating and Denoising with View-guided Tuning in Recommender Systems
>This is our Pytorch implementation for our paper.

## Environment Requirements
 - python==3.7.13
 - pytorch==1.7.1
 - numpy==1.21.6
 -  scipy ==1.7.3
 
## Usage
 1. Configure the SGDVT.conf file in the directory named conf. 
 2.  Run main.py
 >An example conf for ciao
```
training.set=./dataset/ciao/train.txt
test.set=./dataset/ciao/test.txt
social.data=./dataset/ciao/trust.txt
model.name=SGDVT
model.type=graph
item.ranking=-topN 5,10,15,20
embedding.size=64
num.max.epoch=56
batch_size=2048
learnRate=0.001
reg.lambda=0.0001
SGDVT=-n_layer 3 -downthre 5 -upthre 7  -lambda1 0.1 -lambda2 1e-5 -droprate 0.1 -augtype 1 -tau 0.1
output.setup=-dir ./results/.
```

