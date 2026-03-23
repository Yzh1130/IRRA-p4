@echo off
setlocal enabledelayedexpansion

cd /d C:\Users\12823\Desktop\IRRA\IRRA-main
call conda activate irra310
set PYTHONUNBUFFERED=1

echo ==========================================
echo Sweep lambda_unc for MUM + Uncertainty
echo ==========================================

echo.
echo [1/4] lambda_unc = 0.03
python -u train.py ^
  --name exp_rgbt_mum_unc_lunc003_5ep ^
  --dataset_name RGBT-PEDES ^
  --root_dir datasets/TextReID ^
  --img_aug ^
  --batch_size 16 ^
  --test_batch_size 512 ^
  --num_workers 0 ^
  --num_epoch 5 ^
  --warmup_epochs 1 ^
  --log_period 10 ^
  --eval_period 1 ^
  --MLM ^
  --loss_names "sdm+mlm" ^
  --use_mum ^
  --use_global_uncertainty ^
  --uncertainty_dist_type diag_gaussian ^
  --logvar_min -6.0 ^
  --logvar_max 4.0 ^
  --use_precision_fusion ^
  --uncertainty_loss_type w2 ^
  --uncertainty_eps 1e-6 ^
  --lambda_unc 0.03 ^
  --lambda_tr 1.0 ^
  --lambda_rt 1.0 ^
  --lambda_tt 0.0 ^
  --lambda_var_reg 1e-4 ^
  --eval_use_fused_mu

if errorlevel 1 (
  echo [ERROR] lambda_unc=0.03 failed.
  pause
  exit /b 1
)

echo.
echo [2/4] lambda_unc = 0.05
python -u train.py ^
  --name exp_rgbt_mum_unc_lunc005_5ep ^
  --dataset_name RGBT-PEDES ^
  --root_dir datasets/TextReID ^
  --img_aug ^
  --batch_size 16 ^
  --test_batch_size 512 ^
  --num_workers 0 ^
  --num_epoch 5 ^
  --warmup_epochs 1 ^
  --log_period 10 ^
  --eval_period 1 ^
  --MLM ^
  --loss_names "sdm+mlm" ^
  --use_mum ^
  --use_global_uncertainty ^
  --uncertainty_dist_type diag_gaussian ^
  --logvar_min -6.0 ^
  --logvar_max 4.0 ^
  --use_precision_fusion ^
  --uncertainty_loss_type w2 ^
  --uncertainty_eps 1e-6 ^
  --lambda_unc 0.05 ^
  --lambda_tr 1.0 ^
  --lambda_rt 1.0 ^
  --lambda_tt 0.0 ^
  --lambda_var_reg 1e-4 ^
  --eval_use_fused_mu

if errorlevel 1 (
  echo [ERROR] lambda_unc=0.05 failed.
  pause
  exit /b 1
)

echo.
echo [3/4] lambda_unc = 0.08
python -u train.py ^
  --name exp_rgbt_mum_unc_lunc008_5ep ^
  --dataset_name RGBT-PEDES ^
  --root_dir datasets/TextReID ^
  --img_aug ^
  --batch_size 16 ^
  --test_batch_size 512 ^
  --num_workers 0 ^
  --num_epoch 5 ^
  --warmup_epochs 1 ^
  --log_period 10 ^
  --eval_period 1 ^
  --MLM ^
  --loss_names "sdm+mlm" ^
  --use_mum ^
  --use_global_uncertainty ^
  --uncertainty_dist_type diag_gaussian ^
  --logvar_min -6.0 ^
  --logvar_max 4.0 ^
  --use_precision_fusion ^
  --uncertainty_loss_type w2 ^
  --uncertainty_eps 1e-6 ^
  --lambda_unc 0.08 ^
  --lambda_tr 1.0 ^
  --lambda_rt 1.0 ^
  --lambda_tt 0.0 ^
  --lambda_var_reg 1e-4 ^
  --eval_use_fused_mu

if errorlevel 1 (
  echo [ERROR] lambda_unc=0.08 failed.
  pause
  exit /b 1
)

echo.
echo [4/4] lambda_unc = 0.10
python -u train.py ^
  --name exp_rgbt_mum_unc_lunc010_5ep ^
  --dataset_name RGBT-PEDES ^
  --root_dir datasets/TextReID ^
  --img_aug ^
  --batch_size 16 ^
  --test_batch_size 512 ^
  --num_workers 0 ^
  --num_epoch 5 ^
  --warmup_epochs 1 ^
  --log_period 10 ^
  --eval_period 1 ^
  --MLM ^
  --loss_names "sdm+mlm" ^
  --use_mum ^
  --use_global_uncertainty ^
  --uncertainty_dist_type diag_gaussian ^
  --logvar_min -6.0 ^
  --logvar_max 4.0 ^
  --use_precision_fusion ^
  --uncertainty_loss_type w2 ^
  --uncertainty_eps 1e-6 ^
  --lambda_unc 0.10 ^
  --lambda_tr 1.0 ^
  --lambda_rt 1.0 ^
  --lambda_tt 0.0 ^
  --lambda_var_reg 1e-4 ^
  --eval_use_fused_mu

if errorlevel 1 (
  echo [ERROR] lambda_unc=0.10 failed.
  pause
  exit /b 1
)

echo.
echo ==========================================
echo Sweep finished.
echo ==========================================
pause