@echo off
cd /d C:\Users\12823\Desktop\IRRA\IRRA-main

echo ==========================================
echo Start 60-epoch comparison experiments
echo ==========================================

python --version
where python

echo.
echo ==========================================
echo Experiment 1: Baseline 60 epoch
echo ==========================================
python train.py --name exp_rgbt_baseline_60epoch_detcmp --dataset_name RGBT-PEDES --root_dir datasets/TextReID --img_aug --batch_size 16 --MLM --loss_names "sdm+mlm" --num_epoch 60 --warmup_epochs 1 --use_rgbt_add_fusion --lambda_mlm 1.0

echo.
echo ==========================================
echo Experiment 2: Pure MUM 60 epoch (lambda_det=0)
echo ==========================================
python train.py --name exp_rgbt_fullmum_prob_s03_o075_det0_60epoch --dataset_name RGBT-PEDES --root_dir datasets/TextReID --img_aug --batch_size 16 --MLM --loss_names "sdm+mlm" --num_epoch 60 --warmup_epochs 1 --use_rgbt_add_fusion --use_mum --use_batch_uncertainty --use_id_uncertainty --use_prob_sampling --omega 0.75 --uncertainty_scale 0.3 --memory_bank_size 4096 --lambda_mlm 1.0 --lambda_det 0

echo.
echo ==========================================
echo Experiment 3: MUM + deterministic anchor 60 epoch (lambda_det=0.5)
echo ==========================================
python train.py --name exp_rgbt_fullmum_prob_s03_o075_det05_60epoch --dataset_name RGBT-PEDES --root_dir datasets/TextReID --img_aug --batch_size 16 --MLM --loss_names "sdm+mlm" --num_epoch 60 --warmup_epochs 1 --use_rgbt_add_fusion --use_mum --use_batch_uncertainty --use_id_uncertainty --use_prob_sampling --omega 0.75 --uncertainty_scale 0.3 --memory_bank_size 4096 --lambda_mlm 1.0 --lambda_det 0.5

echo.
echo ==========================================
echo All 60-epoch comparison experiments finished
echo ==========================================
pause