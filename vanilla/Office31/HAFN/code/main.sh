#!/bin/bash

task=('A->W' 'D->W' 'W->D' 'A->D' 'D->A' 'W->A')
source=('amazon' 'dslr' 'webcam' 'amazon' 'dslr' 'webcam')
target=('webcam' 'webcam' 'dslr' 'dslr' 'amazon' 'amazon')

post='1'
repeat='1'
data_root='/home/yuancli/projects/fine-tune-domain/data/OFFICE31/'
snapshot='/home/yuancli/projects/fine-tune-domain/product_result/noprivacy/snapshot/'
result='/home/yuancli/projects/fine-tune-domain/product_result/noprivacy/'
epoch=100
gpu_id='0'

#for((index=0; index < 6; index++))
#do
echo ">> traning task "

CUDA_VISIBLE_DEVICES=${gpu_id} python3 train.py \
   --data_root ${data_root} \
   --snapshot ${snapshot} \
   --task product \
   --source mixed_data_Product \
   --target Product_test \
   --epoch ${epoch} \
   --post ${post} \
   --repeat ${repeat}

echo ">> testing task "

CUDA_VISIBLE_DEVICES=${gpu_id} python3 eval.py \
    --data_root ${data_root} \
    --snapshot ${snapshot} \
    --result ${result} \
    --task product \
    --target Product_test \
    --epoch ${epoch} \
   --post ${post} \
   --repeat ${repeat}
#done
