name="FSRA"
data_dir="/mnt/sda/mawenhui/VPRcode/SJJ/U1652/train"
test_dir="/mnt/sda/mawenhui/VPRcode/SJJ/U1652/test"
pretrain_path="/mnt/sda/mawenhui/VPRcode/FSRA/pretrain_model/dinov2_vitb14_reg4_pretrain.pth"
gpu_ids=7
num_worker=4
lr=0.01
sample_num=1
block=3
batchsize=8
triplet_loss=0.3
num_epochs=120
pad=0
views=2
backbone=DINOV2

python train.py --name $name --data_dir $data_dir --gpu_ids $gpu_ids --num_worker $num_worker --views $views --lr $lr \
--sample_num $sample_num --block $block --batchsize $batchsize --triplet_loss $triplet_loss --num_epochs $num_epochs --pretrain_path $pretrain_path\
'''
cd checkpoints/$name
for((i=119;i<=$num_epochs;i+=10));
do
  for((p = 0;p<=$pad;p+=10));
  do
    for ((j = 1; j < 3; j++));
    do
        python test_server.py --test_dir $test_dir --checkpoint net_$i.pth --mode $j --gpu_ids $gpu_ids --num_worker $num_worker --pad $pad
    done
  done
done
'''

