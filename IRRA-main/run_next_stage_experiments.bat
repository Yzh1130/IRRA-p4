@echo off
echo ===============================
echo Stage 1: Baseline 20 epoch
echo ===============================
python train.py --name exp_rgbt_baseline_20epoch --dataset_name RGBT-PEDES --root_dir datasets/TextReID --img_aug --batch_size 16 --MLM --loss_names "sdm+mlm" --num_epoch 20 --warmup_epochs 1 --use_rgbt_add_fusion --lambda_mlm 1.0

echo ===============================
echo Stage 2: Full MUM + sampling + scale=0.3, omega=0.5
echo ===============================
python train.py --name exp_rgbt_fullmum_prob_s03_20epoch --dataset_name RGBT-PEDES --root_dir datasets/TextReID --img_aug --batch_size 16 --MLM --loss_names "sdm+mlm" --num_epoch 20 --warmup_epochs 1 --use_rgbt_add_fusion --use_mum --use_batch_uncertainty --use_id_uncertainty --use_prob_sampling --omega 0.5 --uncertainty_scale 0.3 --memory_bank_size 4096 --lambda_mlm 1.0

echo ===============================
echo Stage 3: Omega ablation 0.75
echo ===============================
python train.py --name exp_rgbt_fullmum_prob_s03_o075_20epoch --dataset_name RGBT-PEDES --root_dir datasets/TextReID --img_aug --batch_size 16 --MLM --loss_names "sdm+mlm" --num_epoch 20 --warmup_epochs 1 --use_rgbt_add_fusion --use_mum --use_batch_uncertainty --use_id_uncertainty --use_prob_sampling --omega 0.75 --uncertainty_scale 0.3 --memory_bank_size 4096 --lambda_mlm 1.0

echo ===============================
echo Stage 4: Omega ablation 0.25
echo ===============================
python train.py --name exp_rgbt_fullmum_prob_s03_o025_20epoch --dataset_name RGBT-PEDES --root_dir datasets/TextReID --img_aug --batch_size 16 --MLM --loss_names "sdm+mlm" --num_epoch 20 --warmup_epochs 1 --use_rgbt_add_fusion --use_mum --use_batch_uncertainty --use_id_uncertainty --use_prob_sampling --omega 0.25 --uncertainty_scale 0.3 --memory_bank_size 4096 --lambda_mlm 1.0

echo ===============================
echo All experiments finished
echo ===============================
pause