# SGDVT: Social Generating and Denoising with View-guided Tuning in Recommender Systems
>This is our Pytorch implementation for our paper.


## Environment Requirements
 - python==3.7.13
 - pytorch==1.7.1
 - numpy==1.21.6
 -  scipy ==1.7.3
 
## Usage
 1. Configure the IDVT.conf file in the directory named conf. 
 2.  Run main.py
 >An example conf for flickr
```
training.set=./dataset/flickr/train.txt
test.set=./dataset/flickr/test.txt
social.data=./dataset/flickr/trust.txt
model.name=IDVT
model.type=graph
item.ranking=-topN 5,10,15,20
embedding.size=64
num.max.epoch=61
batch_size=2048
learnRate=0.001
reg.lambda=0.0001
IDVT=-n_layer 3 -lambda1 0.01 -lambda2 0.001 -droprate 0.2 -augtype 1 -temp1 0.05 -temp2 0.05
output.setup=-dir ./results/.
```

