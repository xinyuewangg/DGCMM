NGPUS=3
TAG=sample_run
DATASET=ChestXray
python train.py ${DATASET} \
--datadir=./datasets/${DATASET} \
--logdir=./logs/${DATASET}/${TAG} \
--arch=resnet_vae3 \
--epoch=100 \
--batch-size=180 \
--lr=5e-3 \
--wd=1e-4 \
--worker=8 \
--T-max=20 \
--h-dim=2048 \
--z-dim=2048 \
--ngpus=${NGPUS} \
--dist-url="tcp://127.0.0.1:53419" \
--multiprocessing-distributed \
--world-size=1 \
--rank=0 \
--weight-cls=10 \
--weight-reconst=0.1 \
--weight-gmm=0.001 \
--pretrained

python augment.py ${DATASET} \
--datadir=./datasets/${DATASET} \
--logdir=./logs/${DATASET}/${TAG} \
--arch=resnet_vae3 \
--epoch=100 \
--batch-size=100 \
--lr=1e-2 \
--wd=1e-4 \
--print-freq=100 \
--worker=8 \
--gpu=0 \
--lr-step=30 \
--h-dim=2048 \
--z-dim=2048 \
--resume=./logs/${DATASET}/${TAG}/checkpoint.pth

python augment_ftl.py ${DATASET} \
--datadir=./datasets/${DATASET} \
--logdir=./logs/${DATASET}/${TAG} \
--arch=resnet_vae3 \
--epoch=100 \
--batch-size=100 \
--lr=1e-2 \
--wd=1e-4 \
--print-freq=100 \
--worker=8 \
--gpu=0 \
--lr-step=30 \
--h-dim=2048 \
--z-dim=2048 \
--resume=./logs/${DATASET}/${TAG}/checkpoint.pth

