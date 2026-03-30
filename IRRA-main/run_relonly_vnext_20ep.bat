@echo off
setlocal enabledelayedexpansion

cd /d C:\Users\12823\Desktop\IRRA\IRRA-main

echo ==========================================
echo Start 4-group 20-epoch local ablation
echo Working dir: %cd%
echo ==========================================

if not exist local_ablation_logs (
    mkdir local_ablation_logs
)

set COMMON_ARGS=--dataset_name RGBT-PEDES --root_dir datasets/TextReID --batch_size 16 --test_batch_size 512 --num_workers 8 --num_epoch 20 --warmup_epochs 1 --log_period 100 --eval_period 1 --MLM --loss_names sdm+mlm --lambda_mlm 1.0 --img_aug --use_rgbt_add_fusion --use_global_uncertainty --uncertainty_dist_type diag_gaussian --use_precision_fusion --uncertainty_loss_type w2 --lambda_unc 0.09 --lambda_tr 1.10 --lambda_rt 0.90 --lambda_tt 0.0 --lambda_var_reg 1e-4 --eval_use_fused_mu

echo.
echo ==========================================
echo [1/4] baseline_20ep_ctrl
echo ==========================================
python train.py --name baseline_20ep_ctrl %COMMON_ARGS% > local_ablation_logs\baseline_20ep_ctrl.log 2>&1
if errorlevel 1 (
    echo baseline_20ep_ctrl FAILED
    pause
    exit /b 1
)

echo.
echo ==========================================
echo [2/4] relonly_gamma_008_20ep
echo ==========================================
python train.py --name relonly_gamma_008_20ep %COMMON_ARGS% --use_image_local_filter --img_local_unc_lambda 0.0 --img_local_fuse_gamma 0.08 > local_ablation_logs\relonly_gamma_008_20ep.log 2>&1
if errorlevel 1 (
    echo relonly_gamma_008_20ep FAILED
    pause
    exit /b 1
)

echo.
echo ==========================================
echo [3/4] relonly_gamma_005_20ep
echo ==========================================
python train.py --name relonly_gamma_005_20ep %COMMON_ARGS% --use_image_local_filter --img_local_unc_lambda 0.0 --img_local_fuse_gamma 0.05 > local_ablation_logs\relonly_gamma_005_20ep.log 2>&1
if errorlevel 1 (
    echo relonly_gamma_005_20ep FAILED
    pause
    exit /b 1
)

echo.
echo ==========================================
echo [4/4] relonly_gamma_003_20ep
echo ==========================================
python train.py --name relonly_gamma_003_20ep %COMMON_ARGS% --use_image_local_filter --img_local_unc_lambda 0.0 --img_local_fuse_gamma 0.03 > local_ablation_logs\relonly_gamma_003_20ep.log 2>&1
if errorlevel 1 (
    echo relonly_gamma_003_20ep FAILED
    pause
    exit /b 1
)

echo.
echo ==========================================
echo All 4 experiments finished successfully.
echo Logs are saved in local_ablation_logs
echo ==========================================
pause