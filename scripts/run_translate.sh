#!/bin/bash

declare -a arr=("A-OKVQA" "CIFAR_100" "CIRR" "FashionIQ" "ImageNet-A" "ImageNet-R" "ImageNet_1K" "MSCOCO" "MSCOCO_i2t" "MSCOCO_t2i" "N24News" "NIGHTS" "OK-VQA" "SUN397" "VOC2007" "Visual7W" "Visual7W-pointing" "VisualNews_i2t" "VisualNews_t2i")
declare -a langs=("it" "fr" "es" "de")

for lang in "${langs[@]}"
do
    for dataset in "${arr[@]}"
    do
        singularity run --nv translator.sif "PYTHONHASHSEED=0 NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 accelerate launch --num_processes=8 translate_instruct.py --dataset_tag ${dataset}  --dataset_split train --translator_name madlad --src_lang en --tgt_lang ${lang}" >> log.txt
    done
done

singularity run --nv translator.sif "PYTHONHASHSEED=0 NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 accelerate launch --num_processes=8 translate_instruct.py --dataset_tag WebQA  --dataset_split train --translator_name madlad --src_lang en --tgt_lang it" >> log.txt
singularity run --nv translator.sif "PYTHONHASHSEED=0 NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 accelerate launch --num_processes=8 translate_instruct.py --dataset_tag WebQA  --dataset_split train --translator_name madlad --src_lang en --tgt_lang es" >> log.txt
singularity run --nv translator.sif "PYTHONHASHSEED=0 NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 accelerate launch --num_processes=8 translate_instruct.py --dataset_tag WebQA  --dataset_split train --translator_name madlad --src_lang en --tgt_lang de" >> log.txt
singularity run --nv translator.sif "PYTHONHASHSEED=0 NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 accelerate launch --num_processes=8 translate_instruct.py --dataset_tag WebQA  --dataset_split train --translator_name madlad --src_lang en --tgt_lang fr" >> log.txt
