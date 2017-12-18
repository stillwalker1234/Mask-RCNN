python ./faster_rcnn/train_net.py \
--gpu 0 \
--iters $3 \
--weights ./data/pretrain_model/Resnet50.npy \
--imdb coco_2014_trainplus35kval \
--cfg  $1 \
--network FPN_train \
--restore $2 \
 
