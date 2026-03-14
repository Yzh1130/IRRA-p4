#!/bin/bash
# Minimal runnable templates for Text-RGBT + MUM ablation experiments.
# Keep argument names aligned with utils/options.py.
#
# Usage:
# 1) Set DATA_ROOT to your dataset root (contains RGBT-PEDES folder).
# 2) Copy one command block below and run it.
#
# Common assumptions:
# - dataset_name: RGBT-PEDES
# - global branch loss: SDM
# - local branch loss: MLM

DATA_ROOT="datasets/TextReID"
DATASET_NAME="RGBT-PEDES"

# ============================================================
# 1) IRRA-RGBT baseline (no MUM)
# Purpose: RGB-T add fusion baseline without uncertainty modeling.
# Key switches:
#   --use_mum (not set) -> False
#   --use_rgbt_add_fusion -> True
#   --use_prob_sampling (not set) -> False
# ============================================================
# CUDA_VISIBLE_DEVICES=0 python train.py \
#   --name exp1_rgbt_baseline_no_mum \
#   --dataset_name ${DATASET_NAME} \
#   --root_dir ${DATA_ROOT} \
#   --img_aug \
#   --batch_size 64 \
#   --MLM \
#   --loss_names 'sdm+mlm' \
#   --num_epoch 60 \
#   --use_rgbt_add_fusion \
#   --lambda_mlm 1.0


# ============================================================
# 2) batch-only uncertainty
# Purpose: only batch-level uncertainty contributes to unified sigma.
# Key switches:
#   --use_mum
#   --use_batch_uncertainty
#   --no_use_id_uncertainty
# ============================================================
# CUDA_VISIBLE_DEVICES=0 python train.py \
#   --name exp2_batch_only_um \
#   --dataset_name ${DATASET_NAME} \
#   --root_dir ${DATA_ROOT} \
#   --img_aug \
#   --batch_size 64 \
#   --MLM \
#   --loss_names 'sdm+mlm' \
#   --num_epoch 60 \
#   --use_rgbt_add_fusion \
#   --use_mum \
#   --use_batch_uncertainty \
#   --no_use_id_uncertainty \
#   --omega 0.5 \
#   --uncertainty_scale 1.0 \
#   --memory_bank_size 0 \
#   --lambda_mlm 1.0


# ============================================================
# 3) id-only uncertainty
# Purpose: only identity-level uncertainty (memory bank) contributes.
# Key switches:
#   --use_mum
#   --no_use_batch_uncertainty
#   --use_id_uncertainty
# ============================================================
# CUDA_VISIBLE_DEVICES=0 python train.py \
#   --name exp3_id_only_um \
#   --dataset_name ${DATASET_NAME} \
#   --root_dir ${DATA_ROOT} \
#   --img_aug \
#   --batch_size 64 \
#   --MLM \
#   --loss_names 'sdm+mlm' \
#   --num_epoch 60 \
#   --use_rgbt_add_fusion \
#   --use_mum \
#   --no_use_batch_uncertainty \
#   --use_id_uncertainty \
#   --omega 0.5 \
#   --uncertainty_scale 1.0 \
#   --memory_bank_size 0 \
#   --lambda_mlm 1.0


# ============================================================
# 4) full MUM
# Purpose: use both batch-level and id-level uncertainty.
# Key switches:
#   --use_mum
#   --use_batch_uncertainty
#   --use_id_uncertainty
# ============================================================
# CUDA_VISIBLE_DEVICES=0 python train.py \
#   --name exp4_full_mum \
#   --dataset_name ${DATASET_NAME} \
#   --root_dir ${DATA_ROOT} \
#   --img_aug \
#   --batch_size 64 \
#   --MLM \
#   --loss_names 'sdm+mlm' \
#   --num_epoch 60 \
#   --use_rgbt_add_fusion \
#   --use_mum \
#   --use_batch_uncertainty \
#   --use_id_uncertainty \
#   --omega 0.5 \
#   --uncertainty_scale 1.0 \
#   --memory_bank_size 0 \
#   --lambda_mlm 1.0


# ============================================================
# 5) deterministic training
# Purpose: compare against probabilistic training while keeping MUM terms.
# Key switches:
#   --use_mum
#   --no_use_prob_sampling
# ============================================================
# CUDA_VISIBLE_DEVICES=0 python train.py \
#   --name exp5_det_training \
#   --dataset_name ${DATASET_NAME} \
#   --root_dir ${DATA_ROOT} \
#   --img_aug \
#   --batch_size 64 \
#   --MLM \
#   --loss_names 'sdm+mlm' \
#   --num_epoch 60 \
#   --use_rgbt_add_fusion \
#   --use_mum \
#   --use_batch_uncertainty \
#   --use_id_uncertainty \
#   --no_use_prob_sampling \
#   --omega 0.5 \
#   --uncertainty_scale 1.0 \
#   --memory_bank_size 0 \
#   --lambda_mlm 1.0


# ============================================================
# 6) probabilistic training
# Purpose: enable reparameterization sampling during training/evaluation.
# Key switches:
#   --use_mum
#   --use_prob_sampling
# ============================================================
# CUDA_VISIBLE_DEVICES=0 python train.py \
#   --name exp6_prob_training \
#   --dataset_name ${DATASET_NAME} \
#   --root_dir ${DATA_ROOT} \
#   --img_aug \
#   --batch_size 64 \
#   --MLM \
#   --loss_names 'sdm+mlm' \
#   --num_epoch 60 \
#   --use_rgbt_add_fusion \
#   --use_mum \
#   --use_batch_uncertainty \
#   --use_id_uncertainty \
#   --use_prob_sampling \
#   --omega 0.5 \
#   --uncertainty_scale 1.0 \
#   --memory_bank_size 0 \
#   --lambda_mlm 1.0
