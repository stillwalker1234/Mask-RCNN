python ./faster_rcnn/test_net.py \
--gpu 0 \
--weights $1 \
--imdb $2 \
--cfg  $3 \
--network FPN_test \
--wait False
