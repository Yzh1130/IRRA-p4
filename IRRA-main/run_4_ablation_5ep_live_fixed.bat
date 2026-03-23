@echo off
setlocal enabledelayedexpansion

echo ==========================================
echo Start 4 local live experiments
echo Output will be shown directly in CMD
echo ==========================================

REM 1) 进入项目目录
cd /d C:\Users\12823\Desktop\IRRA\IRRA-main

REM 2) 激活环境
call conda activate irra310
if errorlevel 1 (
    echo [ERROR] Failed to activate conda env: irra310
    pause
    exit /b 1
)

REM 3) 关闭 Python 缓冲，保证日志实时显示
set PYTHONUNBUFFERED=1

REM 4) GPU 检查
python -c "import torch; print('cuda_available =', torch.cuda.is_available()); print('gpu_count =', torch.cuda.device_count())"
if errorlevel 1 (
    echo [ERROR] PyTorch check failed.
    pause
    exit /b 1
)

REM 5) 语法检查
python -m py_compile utils\options.py model\build.py model\objectives.py processor\processor.py
if errorlevel 1 (
    echo [ERROR] Python compile check failed.
    pause
    exit /b 1
)

echo.
echo ==========================================
echo [1/4] Baseline: sdm+mlm
echo ==========================================
python -u train.py ^
  --name exp_rgbt_baseline_5epoch_live ^
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
  --loss_names "sdm+mlm"

if errorlevel 1 (
    echo [ERROR] Experiment 1 failed.
    pause
    exit /b 1
)

echo.
echo ==========================================
echo [2/4] MUM only
echo ==========================================
python -u train.py ^
  --name exp_rgbt_mum_only_5epoch_live ^
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
  --use_mum

if errorlevel 1 (
    echo [ERROR] Experiment 2 failed.
    pause
    exit /b 1
)

echo.
echo ==========================================
echo [3/4] Uncertainty only
echo ==========================================
python -u train.py ^
  --name exp_rgbt_unc_only_5epoch_live ^
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
  --use_global_uncertainty ^
  --uncertainty_dist_type diag_gaussian ^
  --logvar_min -6.0 ^
  --logvar_max 4.0 ^
  --use_precision_fusion ^
  --uncertainty_loss_type w2 ^
  --uncertainty_eps 1e-6 ^
  --lambda_unc 0.1 ^
  --lambda_tr 1.0 ^
  --lambda_rt 1.0 ^
  --lambda_tt 0.0 ^
  --lambda_var_reg 1e-4 ^
  --eval_use_fused_mu

if errorlevel 1 (
    echo [ERROR] Experiment 3 failed.
    pause
    exit /b 1
)

echo.
echo ==========================================
echo [4/4] MUM + Uncertainty
echo ==========================================
python -u train.py ^
  --name exp_rgbt_mum_plus_unc_5epoch_live ^
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
  --lambda_unc 0.1 ^
  --lambda_tr 1.0 ^
  --lambda_rt 1.0 ^
  --lambda_tt 0.0 ^
  --lambda_var_reg 1e-4 ^
  --eval_use_fused_mu

if errorlevel 1 (
    echo [ERROR] Experiment 4 failed.
    pause
    exit /b 1
)

echo.
echo ==========================================
echo All 4 experiments finished successfully.
echo ==========================================
pause