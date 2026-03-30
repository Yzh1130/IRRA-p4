@echo off
setlocal

cd /d C:\Users\12823\Desktop\IRRA\IRRA-main

echo ==========================================
echo Start gamma-schedule 20-epoch ablation
echo Real-time logs will be shown on screen
echo Logs will also be saved to files
echo Working dir: %cd%
echo ==========================================

if not exist gamma_schedule_logs (
    mkdir gamma_schedule_logs
)

set COMMON_ARGS=--dataset_name RGBT-PEDES --root_dir datasets/TextReID --batch_size 16 --test_batch_size 512 --num_workers 8 --num_epoch 20 --warmup_epochs 1 --log_period 100 --eval_period 1 --MLM --loss_names sdm+mlm --lambda_mlm 1.0 --img_aug --use_rgbt_add_fusion --use_global_uncertainty --uncertainty_dist_type diag_gaussian --use_precision_fusion --uncertainty_loss_type w2 --lambda_unc 0.09 --lambda_tr 1.10 --lambda_rt 0.90 --lambda_tt 0.0 --lambda_var_reg 1e-4 --eval_use_fused_mu

echo.
echo ==========================================
echo [1/4] baseline_20ep_ctrl
echo ==========================================
powershell -NoProfile -Command "python -u train.py --name baseline_20ep_ctrl %COMMON_ARGS% 2>&1 | Tee-Object -FilePath 'gamma_schedule_logs\baseline_20ep_ctrl.log'"
if errorlevel 1 (
    echo baseline_20ep_ctrl FAILED
    pause
    exit /b 1
)

echo.
echo ==========================================
echo [2/4] relonly_gamma005_always_20ep
echo ==========================================
powershell -NoProfile -Command "python -u train.py --name relonly_gamma005_always_20ep %COMMON_ARGS% --use_image_local_filter --img_local_unc_lambda 0.0 --img_local_fuse_gamma 0.05 --img_local_stage_mode always --img_local_gamma_schedule_mode constant 2>&1 | Tee-Object -FilePath 'gamma_schedule_logs\relonly_gamma005_always_20ep.log'"
if errorlevel 1 (
    echo relonly_gamma005_always_20ep FAILED
    pause
    exit /b 1
)

echo.
echo ==========================================
echo [3/4] stage_relonly_gamma005_early_1to8_20ep
echo ==========================================
powershell -NoProfile -Command "python -u train.py --name stage_relonly_gamma005_early_1to8_20ep %COMMON_ARGS% --use_image_local_filter --img_local_unc_lambda 0.0 --img_local_fuse_gamma 0.05 --img_local_stage_mode window --img_local_stage_start 1 --img_local_stage_end 8 --img_local_gamma_schedule_mode constant 2>&1 | Tee-Object -FilePath 'gamma_schedule_logs\stage_relonly_gamma005_early_1to8_20ep.log'"
if errorlevel 1 (
    echo stage_relonly_gamma005_early_1to8_20ep FAILED
    pause
    exit /b 1
)

echo.
echo ==========================================
echo [4/4] stage_relonly_gamma_schedule_20ep
echo ==========================================
powershell -NoProfile -Command "python -u train.py --name stage_relonly_gamma_schedule_20ep %COMMON_ARGS% --use_image_local_filter --img_local_unc_lambda 0.0 --img_local_stage_mode always --img_local_gamma_schedule_mode three_stage --img_local_gamma_stage1_end 6 --img_local_gamma_stage2_end 12 --img_local_gamma_stage1_value 0.05 --img_local_gamma_stage2_value 0.02 --img_local_gamma_stage3_value 0.0 2>&1 | Tee-Object -FilePath 'gamma_schedule_logs\stage_relonly_gamma_schedule_20ep.log'"
if errorlevel 1 (
    echo stage_relonly_gamma_schedule_20ep FAILED
    pause
    exit /b 1
)

echo.
echo ==========================================
echo All 4 experiments finished successfully.
echo Logs are saved in gamma_schedule_logs
echo ==========================================
pause