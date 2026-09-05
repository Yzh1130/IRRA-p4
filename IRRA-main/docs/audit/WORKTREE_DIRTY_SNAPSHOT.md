# Worktree Dirty Snapshot

该文件是一次性事实快照，仅描述采集时 Git porcelain 与逐文件 diff --stat 的输出。

## 一、快照元数据

| 检查项 | 值 | 状态 | 来源 | 备注 |
| --- | --- | --- | --- | --- |
| 采集时间 | 2026-09-05T17:16:04+08:00 | OK | Python datetime.now().astimezone() |  |
| Git 顶层目录 | E:\论文2修改\IRRA\IRRA-main-mraf | OK | git rev-parse --show-toplevel 的采集结果 |  |
| 快照记录数 | 13 | WARN | git status --porcelain=v1 -z --untracked-files=all（repo=E:\论文2修改\IRRA\IRRA-main-mraf） | 报告与原始 JSON 均在本次状态采集之后写入 |
| Stage 0 前既有 tracked EOL 基线（合计） | 118 | WARN | git ls-files -z（repo=E:\论文2修改\IRRA\IRRA-main-mraf）; 对每个候选执行 git show HEAD:<path>，并以 UTF-8 读取工作树后比较 CRLF/LF 归一化结果 | 仅 EOL 差异 + `.gitignore` 的 EOL 差异及本任务允许追加行；不会暂存这些换行改写 |
| tracked 文本仅 EOL 差异 | 118 | WARN | git ls-files -z（repo=E:\论文2修改\IRRA\IRRA-main-mraf）; 对每个候选执行 git show HEAD:<path>，并以 UTF-8 读取工作树后比较 CRLF/LF 归一化结果 | IRRA-main/.idea/.gitignore；IRRA-main/.idea/IRRA-main.iml；IRRA-main/.idea/inspectionProfiles/profiles_settings.xml；IRRA-main/.idea/misc.xml；IRRA-main/.idea/modules.xml；IRRA-main/LICENSE；IRRA-main/README.md；IRRA-main/datasets/__init__.py；IRRA-main/datasets/bases.py；IRRA-main/datasets/build.py；IRRA-main/datasets/cuhkpedes.py；IRRA-main/datasets/icfgpedes.py；IRRA-main/datasets/mga_pseudo.py；IRRA-main/datasets/preprocessing.py；IRRA-main/datasets/rgbtpedes.py；IRRA-main/datasets/rstpreid.py；IRRA-main/datasets/sampler.py；IRRA-main/datasets/sampler_ddp.py；IRRA-main/docs/MGA_RDE_USAGE.md；IRRA-main/docs/revision/degradation_implementation_facts.md；IRRA-main/docs/revision/local_validation_report.md；IRRA-main/eval_local.py；IRRA-main/model/__init__.py；IRRA-main/model/clip_model.py；IRRA-main/model/objectives.py；IRRA-main/model/prototype_bank.py；IRRA-main/model/rgbt_moe_adapter.py；IRRA-main/model/text_guided_masking.py；IRRA-main/model/uncertainty.py；IRRA-main/pack/20260420_094400_synthv5_rollout/banks/synth_caption_bank_smoketest.json；IRRA-main/pack/20260420_094400_synthv5_rollout/vllm_pid.txt；IRRA-main/pack/20260420_094400_synthv5_rollout/vllm_port.txt；IRRA-main/pack/20260428_193242_hardneg_v2_sweep/run_hardneg_v2_all.sh；IRRA-main/pack/20260428_193242_hardneg_v2_sweep/summarize_hardneg_v2.py；IRRA-main/pack/20260429_183453_proto_calib_v2_sweep/COMMANDS.txt；IRRA-main/pack/20260429_183453_proto_calib_v2_sweep/run_proto_calib_v2_all.sh；IRRA-main/pack/20260429_183453_proto_calib_v2_sweep/summarize_proto_calib_v2.py；IRRA-main/processor/__init__.py；IRRA-main/processor/processor.py；IRRA-main/run_4_ablation_5ep_live_fixed.bat；IRRA-main/run_det_anchor_60epoch.bat；IRRA-main/run_gamma_schedule_20ep_realtime.bat；IRRA-main/run_hn_60ep_4exp.sh；IRRA-main/run_irra.sh；IRRA-main/run_mga_rde_screening_4exp_20ep.sh；IRRA-main/run_next_stage_experiments.bat；IRRA-main/run_relonly_vnext_20ep.bat；IRRA-main/run_relonly_vnext_20ep_realtime.bat；IRRA-main/run_rgbt_mum_ablations.sh；IRRA-main/run_stage_relonly_gamma005_20ep_realtime.bat；IRRA-main/run_strict_proto3_uspc_20ep.sh；IRRA-main/run_synthcap_yunwu_20ep_3exp.sh；IRRA-main/run_synthcap_yunwu_smoke.sh；IRRA-main/run_uspc_v1_ablation_60ep.sh；IRRA-main/run_uspc_v1_smoke_5ep.sh；IRRA-main/run_uspc_v2a_attr_smoke_2ep.sh；IRRA-main/run_uspc_v2b0_slot_local_20ep_grid.sh；IRRA-main/run_uspc_v2b0_slot_local_smoke_2ep.sh；IRRA-main/scripts/revision/README_AutoDL.md；IRRA-main/scripts/revision/run_new_degradation_matrix.sh；IRRA-main/scripts/revision/run_old_degradation_regression.sh；IRRA-main/solver/__init__.py；IRRA-main/solver/build.py；IRRA-main/solver/lr_scheduler.py；IRRA-main/sweep_lambda_unc_5ep.bat；IRRA-main/test.py；IRRA-main/tests/__init__.py；IRRA-main/tests/test_evaluation_degradation.py；IRRA-main/tests/test_revision_analysis_tools.py；IRRA-main/tools/analyze_qalr_v3_candidate_oracle.py；IRRA-main/tools/build_large_synth_bank_v2.py；IRRA-main/tools/build_large_synth_bank_v3.py；IRRA-main/tools/build_large_synth_bank_v4.py；IRRA-main/tools/build_large_synth_bank_v5_llm.py；IRRA-main/tools/build_mga_pseudo_from_captions.py；IRRA-main/tools/build_offline_hardneg_index.py；IRRA-main/tools/build_offline_prototype_bank.py；IRRA-main/tools/build_qalr_v4_gate_dataset.py；IRRA-main/tools/build_synth_bank_smoketest.py；IRRA-main/tools/build_synth_bank_v6_controlled.py；IRRA-main/tools/check_synth_bank_v6_controlled.py；IRRA-main/tools/check_synth_caption_bank.py；IRRA-main/tools/dump_rgbt_pedes_test_metadata.py；IRRA-main/tools/ensemble_eval_scores.py；IRRA-main/tools/revision/__init__.py；IRRA-main/tools/revision/color_group_evaluation.py；IRRA-main/tools/revision/color_terms_en.txt；IRRA-main/tools/revision/compare_score_files.py；IRRA-main/tools/revision/ensemble_scores.py；IRRA-main/tools/revision/feature_norm_analysis.py；IRRA-main/tools/revision/intensity_curve_summary.py；IRRA-main/tools/revision/noise_repeat_summary.py；IRRA-main/tools/revision/paired_seed_statistics.py；IRRA-main/tools/revision/score_io.py；IRRA-main/tools/revision/summarize_robustness.py；IRRA-main/tools/run_qalr_v2_safe_grid.sh；IRRA-main/tools/smoke_mga_dataloader.py；IRRA-main/tools/train_qalr_v4_gate.py；IRRA-main/tools/validate_mga_pseudo_json.py；IRRA-main/tools/weight_split_eval.py；IRRA-main/train.py；IRRA-main/utils/__init__.py；IRRA-main/utils/attribute_parser.py；IRRA-main/utils/checkpoint.py；IRRA-main/utils/comm.py；IRRA-main/utils/eval_degradation.py；IRRA-main/utils/iotools.py；IRRA-main/utils/logger.py；IRRA-main/utils/mdrc.py；IRRA-main/utils/meter.py；IRRA-main/utils/metrics.py；IRRA-main/utils/mtp_text.py；IRRA-main/utils/qalr_v4_gate.py；IRRA-main/utils/query_adaptive_rerank.py；IRRA-main/utils/simple_tokenizer.py；IRRA-main/utils/transport_rerank.py；IRRA-main/utils/uspc_text_views.py；IRRA-main/visualize.py |
| EOL 差异且仅追加 audit_out/ | 0 | OK | git ls-files -z（repo=E:\论文2修改\IRRA\IRRA-main-mraf）; 对每个候选执行 git show HEAD:<path>，并以 UTF-8 读取工作树后比较 CRLF/LF 归一化结果 |  |
| 其它 tracked 文本语义差异 | 1 | WARN | git ls-files -z（repo=E:\论文2修改\IRRA\IRRA-main-mraf）; 对每个候选执行 git show HEAD:<path>，并以 UTF-8 读取工作树后比较 CRLF/LF 归一化结果 | .gitignore |

## 二、未提交改动

| XY | 路径 | 原路径 | diff --stat | 状态 | 来源 | 备注 |
| --- | --- | --- | --- | --- | --- | --- |
|  M | .gitignore |  | .gitignore \| 80 ++++++++++++++++++++++++++++++++------------------------------<br> 1 file changed, 41 insertions(+), 39 deletions(-) | WARN | git status --porcelain=v1 -z --untracked-files=all（repo=E:\论文2修改\IRRA\IRRA-main-mraf）; git diff --stat HEAD -- .gitignore（repo=E:\论文2修改\IRRA\IRRA-main-mraf） |  |
| ?? | IRRA-main/docs/audit/CODE_INVENTORY.md |  | 不适用（未跟踪文件） | WARN | git status --porcelain=v1 -z --untracked-files=all（repo=E:\论文2修改\IRRA\IRRA-main-mraf） | git diff 不显示未跟踪文件 |
| ?? | IRRA-main/docs/audit/DATA_AUDIT_REPORT.md |  | 不适用（未跟踪文件） | WARN | git status --porcelain=v1 -z --untracked-files=all（repo=E:\论文2修改\IRRA\IRRA-main-mraf） | git diff 不显示未跟踪文件 |
| ?? | IRRA-main/docs/audit/README.md |  | 不适用（未跟踪文件） | WARN | git status --porcelain=v1 -z --untracked-files=all（repo=E:\论文2修改\IRRA\IRRA-main-mraf） | git diff 不显示未跟踪文件 |
| ?? | IRRA-main/docs/audit/REPO_STATE.md |  | 不适用（未跟踪文件） | WARN | git status --porcelain=v1 -z --untracked-files=all（repo=E:\论文2修改\IRRA\IRRA-main-mraf） | git diff 不显示未跟踪文件 |
| ?? | IRRA-main/docs/audit/WORKTREE_DIRTY_SNAPSHOT.md |  | 不适用（未跟踪文件） | WARN | git status --porcelain=v1 -z --untracked-files=all（repo=E:\论文2修改\IRRA\IRRA-main-mraf） | git diff 不显示未跟踪文件 |
| ?? | IRRA-main/tools/audit/__init__.py |  | 不适用（未跟踪文件） | WARN | git status --porcelain=v1 -z --untracked-files=all（repo=E:\论文2修改\IRRA\IRRA-main-mraf） | git diff 不显示未跟踪文件 |
| ?? | IRRA-main/tools/audit/audit_assets.py |  | 不适用（未跟踪文件） | WARN | git status --porcelain=v1 -z --untracked-files=all（repo=E:\论文2修改\IRRA\IRRA-main-mraf） | git diff 不显示未跟踪文件 |
| ?? | IRRA-main/tools/audit/audit_hparams.py |  | 不适用（未跟踪文件） | WARN | git status --porcelain=v1 -z --untracked-files=all（repo=E:\论文2修改\IRRA\IRRA-main-mraf） | git diff 不显示未跟踪文件 |
| ?? | IRRA-main/tools/audit/audit_repo_state.py |  | 不适用（未跟踪文件） | WARN | git status --porcelain=v1 -z --untracked-files=all（repo=E:\论文2修改\IRRA\IRRA-main-mraf） | git diff 不显示未跟踪文件 |
| ?? | IRRA-main/tools/audit/common.py |  | 不适用（未跟踪文件） | WARN | git status --porcelain=v1 -z --untracked-files=all（repo=E:\论文2修改\IRRA\IRRA-main-mraf） | git diff 不显示未跟踪文件 |
| ?? | IRRA-main/tools/audit/probe_dataset.py |  | 不适用（未跟踪文件） | WARN | git status --porcelain=v1 -z --untracked-files=all（repo=E:\论文2修改\IRRA\IRRA-main-mraf） | git diff 不显示未跟踪文件 |
| ?? | IRRA-main/tools/audit/run_all_audits.sh |  | 不适用（未跟踪文件） | WARN | git status --porcelain=v1 -z --untracked-files=all（repo=E:\论文2修改\IRRA\IRRA-main-mraf） | git diff 不显示未跟踪文件 |

## 三、EOL 内容核验清单

| 类别 | 文件 | source | status |
| --- | --- | --- | --- |
| 仅 EOL | IRRA-main/.idea/.gitignore | git ls-files -z（repo=E:\论文2修改\IRRA\IRRA-main-mraf）; 对每个候选执行 git show HEAD:<path>，并以 UTF-8 读取工作树后比较 CRLF/LF 归一化结果 | WARN |
| 仅 EOL | IRRA-main/.idea/IRRA-main.iml | git ls-files -z（repo=E:\论文2修改\IRRA\IRRA-main-mraf）; 对每个候选执行 git show HEAD:<path>，并以 UTF-8 读取工作树后比较 CRLF/LF 归一化结果 | WARN |
| 仅 EOL | IRRA-main/.idea/inspectionProfiles/profiles_settings.xml | git ls-files -z（repo=E:\论文2修改\IRRA\IRRA-main-mraf）; 对每个候选执行 git show HEAD:<path>，并以 UTF-8 读取工作树后比较 CRLF/LF 归一化结果 | WARN |
| 仅 EOL | IRRA-main/.idea/misc.xml | git ls-files -z（repo=E:\论文2修改\IRRA\IRRA-main-mraf）; 对每个候选执行 git show HEAD:<path>，并以 UTF-8 读取工作树后比较 CRLF/LF 归一化结果 | WARN |
| 仅 EOL | IRRA-main/.idea/modules.xml | git ls-files -z（repo=E:\论文2修改\IRRA\IRRA-main-mraf）; 对每个候选执行 git show HEAD:<path>，并以 UTF-8 读取工作树后比较 CRLF/LF 归一化结果 | WARN |
| 仅 EOL | IRRA-main/LICENSE | git ls-files -z（repo=E:\论文2修改\IRRA\IRRA-main-mraf）; 对每个候选执行 git show HEAD:<path>，并以 UTF-8 读取工作树后比较 CRLF/LF 归一化结果 | WARN |
| 仅 EOL | IRRA-main/README.md | git ls-files -z（repo=E:\论文2修改\IRRA\IRRA-main-mraf）; 对每个候选执行 git show HEAD:<path>，并以 UTF-8 读取工作树后比较 CRLF/LF 归一化结果 | WARN |
| 仅 EOL | IRRA-main/datasets/__init__.py | git ls-files -z（repo=E:\论文2修改\IRRA\IRRA-main-mraf）; 对每个候选执行 git show HEAD:<path>，并以 UTF-8 读取工作树后比较 CRLF/LF 归一化结果 | WARN |
| 仅 EOL | IRRA-main/datasets/bases.py | git ls-files -z（repo=E:\论文2修改\IRRA\IRRA-main-mraf）; 对每个候选执行 git show HEAD:<path>，并以 UTF-8 读取工作树后比较 CRLF/LF 归一化结果 | WARN |
| 仅 EOL | IRRA-main/datasets/build.py | git ls-files -z（repo=E:\论文2修改\IRRA\IRRA-main-mraf）; 对每个候选执行 git show HEAD:<path>，并以 UTF-8 读取工作树后比较 CRLF/LF 归一化结果 | WARN |
| 仅 EOL | IRRA-main/datasets/cuhkpedes.py | git ls-files -z（repo=E:\论文2修改\IRRA\IRRA-main-mraf）; 对每个候选执行 git show HEAD:<path>，并以 UTF-8 读取工作树后比较 CRLF/LF 归一化结果 | WARN |
| 仅 EOL | IRRA-main/datasets/icfgpedes.py | git ls-files -z（repo=E:\论文2修改\IRRA\IRRA-main-mraf）; 对每个候选执行 git show HEAD:<path>，并以 UTF-8 读取工作树后比较 CRLF/LF 归一化结果 | WARN |
| 仅 EOL | IRRA-main/datasets/mga_pseudo.py | git ls-files -z（repo=E:\论文2修改\IRRA\IRRA-main-mraf）; 对每个候选执行 git show HEAD:<path>，并以 UTF-8 读取工作树后比较 CRLF/LF 归一化结果 | WARN |
| 仅 EOL | IRRA-main/datasets/preprocessing.py | git ls-files -z（repo=E:\论文2修改\IRRA\IRRA-main-mraf）; 对每个候选执行 git show HEAD:<path>，并以 UTF-8 读取工作树后比较 CRLF/LF 归一化结果 | WARN |
| 仅 EOL | IRRA-main/datasets/rgbtpedes.py | git ls-files -z（repo=E:\论文2修改\IRRA\IRRA-main-mraf）; 对每个候选执行 git show HEAD:<path>，并以 UTF-8 读取工作树后比较 CRLF/LF 归一化结果 | WARN |
| 仅 EOL | IRRA-main/datasets/rstpreid.py | git ls-files -z（repo=E:\论文2修改\IRRA\IRRA-main-mraf）; 对每个候选执行 git show HEAD:<path>，并以 UTF-8 读取工作树后比较 CRLF/LF 归一化结果 | WARN |
| 仅 EOL | IRRA-main/datasets/sampler.py | git ls-files -z（repo=E:\论文2修改\IRRA\IRRA-main-mraf）; 对每个候选执行 git show HEAD:<path>，并以 UTF-8 读取工作树后比较 CRLF/LF 归一化结果 | WARN |
| 仅 EOL | IRRA-main/datasets/sampler_ddp.py | git ls-files -z（repo=E:\论文2修改\IRRA\IRRA-main-mraf）; 对每个候选执行 git show HEAD:<path>，并以 UTF-8 读取工作树后比较 CRLF/LF 归一化结果 | WARN |
| 仅 EOL | IRRA-main/docs/MGA_RDE_USAGE.md | git ls-files -z（repo=E:\论文2修改\IRRA\IRRA-main-mraf）; 对每个候选执行 git show HEAD:<path>，并以 UTF-8 读取工作树后比较 CRLF/LF 归一化结果 | WARN |
| 仅 EOL | IRRA-main/docs/revision/degradation_implementation_facts.md | git ls-files -z（repo=E:\论文2修改\IRRA\IRRA-main-mraf）; 对每个候选执行 git show HEAD:<path>，并以 UTF-8 读取工作树后比较 CRLF/LF 归一化结果 | WARN |
| 仅 EOL | IRRA-main/docs/revision/local_validation_report.md | git ls-files -z（repo=E:\论文2修改\IRRA\IRRA-main-mraf）; 对每个候选执行 git show HEAD:<path>，并以 UTF-8 读取工作树后比较 CRLF/LF 归一化结果 | WARN |
| 仅 EOL | IRRA-main/eval_local.py | git ls-files -z（repo=E:\论文2修改\IRRA\IRRA-main-mraf）; 对每个候选执行 git show HEAD:<path>，并以 UTF-8 读取工作树后比较 CRLF/LF 归一化结果 | WARN |
| 仅 EOL | IRRA-main/model/__init__.py | git ls-files -z（repo=E:\论文2修改\IRRA\IRRA-main-mraf）; 对每个候选执行 git show HEAD:<path>，并以 UTF-8 读取工作树后比较 CRLF/LF 归一化结果 | WARN |
| 仅 EOL | IRRA-main/model/clip_model.py | git ls-files -z（repo=E:\论文2修改\IRRA\IRRA-main-mraf）; 对每个候选执行 git show HEAD:<path>，并以 UTF-8 读取工作树后比较 CRLF/LF 归一化结果 | WARN |
| 仅 EOL | IRRA-main/model/objectives.py | git ls-files -z（repo=E:\论文2修改\IRRA\IRRA-main-mraf）; 对每个候选执行 git show HEAD:<path>，并以 UTF-8 读取工作树后比较 CRLF/LF 归一化结果 | WARN |
| 仅 EOL | IRRA-main/model/prototype_bank.py | git ls-files -z（repo=E:\论文2修改\IRRA\IRRA-main-mraf）; 对每个候选执行 git show HEAD:<path>，并以 UTF-8 读取工作树后比较 CRLF/LF 归一化结果 | WARN |
| 仅 EOL | IRRA-main/model/rgbt_moe_adapter.py | git ls-files -z（repo=E:\论文2修改\IRRA\IRRA-main-mraf）; 对每个候选执行 git show HEAD:<path>，并以 UTF-8 读取工作树后比较 CRLF/LF 归一化结果 | WARN |
| 仅 EOL | IRRA-main/model/text_guided_masking.py | git ls-files -z（repo=E:\论文2修改\IRRA\IRRA-main-mraf）; 对每个候选执行 git show HEAD:<path>，并以 UTF-8 读取工作树后比较 CRLF/LF 归一化结果 | WARN |
| 仅 EOL | IRRA-main/model/uncertainty.py | git ls-files -z（repo=E:\论文2修改\IRRA\IRRA-main-mraf）; 对每个候选执行 git show HEAD:<path>，并以 UTF-8 读取工作树后比较 CRLF/LF 归一化结果 | WARN |
| 仅 EOL | IRRA-main/pack/20260420_094400_synthv5_rollout/banks/synth_caption_bank_smoketest.json | git ls-files -z（repo=E:\论文2修改\IRRA\IRRA-main-mraf）; 对每个候选执行 git show HEAD:<path>，并以 UTF-8 读取工作树后比较 CRLF/LF 归一化结果 | WARN |
| 仅 EOL | IRRA-main/pack/20260420_094400_synthv5_rollout/vllm_pid.txt | git ls-files -z（repo=E:\论文2修改\IRRA\IRRA-main-mraf）; 对每个候选执行 git show HEAD:<path>，并以 UTF-8 读取工作树后比较 CRLF/LF 归一化结果 | WARN |
| 仅 EOL | IRRA-main/pack/20260420_094400_synthv5_rollout/vllm_port.txt | git ls-files -z（repo=E:\论文2修改\IRRA\IRRA-main-mraf）; 对每个候选执行 git show HEAD:<path>，并以 UTF-8 读取工作树后比较 CRLF/LF 归一化结果 | WARN |
| 仅 EOL | IRRA-main/pack/20260428_193242_hardneg_v2_sweep/run_hardneg_v2_all.sh | git ls-files -z（repo=E:\论文2修改\IRRA\IRRA-main-mraf）; 对每个候选执行 git show HEAD:<path>，并以 UTF-8 读取工作树后比较 CRLF/LF 归一化结果 | WARN |
| 仅 EOL | IRRA-main/pack/20260428_193242_hardneg_v2_sweep/summarize_hardneg_v2.py | git ls-files -z（repo=E:\论文2修改\IRRA\IRRA-main-mraf）; 对每个候选执行 git show HEAD:<path>，并以 UTF-8 读取工作树后比较 CRLF/LF 归一化结果 | WARN |
| 仅 EOL | IRRA-main/pack/20260429_183453_proto_calib_v2_sweep/COMMANDS.txt | git ls-files -z（repo=E:\论文2修改\IRRA\IRRA-main-mraf）; 对每个候选执行 git show HEAD:<path>，并以 UTF-8 读取工作树后比较 CRLF/LF 归一化结果 | WARN |
| 仅 EOL | IRRA-main/pack/20260429_183453_proto_calib_v2_sweep/run_proto_calib_v2_all.sh | git ls-files -z（repo=E:\论文2修改\IRRA\IRRA-main-mraf）; 对每个候选执行 git show HEAD:<path>，并以 UTF-8 读取工作树后比较 CRLF/LF 归一化结果 | WARN |
| 仅 EOL | IRRA-main/pack/20260429_183453_proto_calib_v2_sweep/summarize_proto_calib_v2.py | git ls-files -z（repo=E:\论文2修改\IRRA\IRRA-main-mraf）; 对每个候选执行 git show HEAD:<path>，并以 UTF-8 读取工作树后比较 CRLF/LF 归一化结果 | WARN |
| 仅 EOL | IRRA-main/processor/__init__.py | git ls-files -z（repo=E:\论文2修改\IRRA\IRRA-main-mraf）; 对每个候选执行 git show HEAD:<path>，并以 UTF-8 读取工作树后比较 CRLF/LF 归一化结果 | WARN |
| 仅 EOL | IRRA-main/processor/processor.py | git ls-files -z（repo=E:\论文2修改\IRRA\IRRA-main-mraf）; 对每个候选执行 git show HEAD:<path>，并以 UTF-8 读取工作树后比较 CRLF/LF 归一化结果 | WARN |
| 仅 EOL | IRRA-main/run_4_ablation_5ep_live_fixed.bat | git ls-files -z（repo=E:\论文2修改\IRRA\IRRA-main-mraf）; 对每个候选执行 git show HEAD:<path>，并以 UTF-8 读取工作树后比较 CRLF/LF 归一化结果 | WARN |
| 仅 EOL | IRRA-main/run_det_anchor_60epoch.bat | git ls-files -z（repo=E:\论文2修改\IRRA\IRRA-main-mraf）; 对每个候选执行 git show HEAD:<path>，并以 UTF-8 读取工作树后比较 CRLF/LF 归一化结果 | WARN |
| 仅 EOL | IRRA-main/run_gamma_schedule_20ep_realtime.bat | git ls-files -z（repo=E:\论文2修改\IRRA\IRRA-main-mraf）; 对每个候选执行 git show HEAD:<path>，并以 UTF-8 读取工作树后比较 CRLF/LF 归一化结果 | WARN |
| 仅 EOL | IRRA-main/run_hn_60ep_4exp.sh | git ls-files -z（repo=E:\论文2修改\IRRA\IRRA-main-mraf）; 对每个候选执行 git show HEAD:<path>，并以 UTF-8 读取工作树后比较 CRLF/LF 归一化结果 | WARN |
| 仅 EOL | IRRA-main/run_irra.sh | git ls-files -z（repo=E:\论文2修改\IRRA\IRRA-main-mraf）; 对每个候选执行 git show HEAD:<path>，并以 UTF-8 读取工作树后比较 CRLF/LF 归一化结果 | WARN |
| 仅 EOL | IRRA-main/run_mga_rde_screening_4exp_20ep.sh | git ls-files -z（repo=E:\论文2修改\IRRA\IRRA-main-mraf）; 对每个候选执行 git show HEAD:<path>，并以 UTF-8 读取工作树后比较 CRLF/LF 归一化结果 | WARN |
| 仅 EOL | IRRA-main/run_next_stage_experiments.bat | git ls-files -z（repo=E:\论文2修改\IRRA\IRRA-main-mraf）; 对每个候选执行 git show HEAD:<path>，并以 UTF-8 读取工作树后比较 CRLF/LF 归一化结果 | WARN |
| 仅 EOL | IRRA-main/run_relonly_vnext_20ep.bat | git ls-files -z（repo=E:\论文2修改\IRRA\IRRA-main-mraf）; 对每个候选执行 git show HEAD:<path>，并以 UTF-8 读取工作树后比较 CRLF/LF 归一化结果 | WARN |
| 仅 EOL | IRRA-main/run_relonly_vnext_20ep_realtime.bat | git ls-files -z（repo=E:\论文2修改\IRRA\IRRA-main-mraf）; 对每个候选执行 git show HEAD:<path>，并以 UTF-8 读取工作树后比较 CRLF/LF 归一化结果 | WARN |
| 仅 EOL | IRRA-main/run_rgbt_mum_ablations.sh | git ls-files -z（repo=E:\论文2修改\IRRA\IRRA-main-mraf）; 对每个候选执行 git show HEAD:<path>，并以 UTF-8 读取工作树后比较 CRLF/LF 归一化结果 | WARN |
| 仅 EOL | IRRA-main/run_stage_relonly_gamma005_20ep_realtime.bat | git ls-files -z（repo=E:\论文2修改\IRRA\IRRA-main-mraf）; 对每个候选执行 git show HEAD:<path>，并以 UTF-8 读取工作树后比较 CRLF/LF 归一化结果 | WARN |
| 仅 EOL | IRRA-main/run_strict_proto3_uspc_20ep.sh | git ls-files -z（repo=E:\论文2修改\IRRA\IRRA-main-mraf）; 对每个候选执行 git show HEAD:<path>，并以 UTF-8 读取工作树后比较 CRLF/LF 归一化结果 | WARN |
| 仅 EOL | IRRA-main/run_synthcap_yunwu_20ep_3exp.sh | git ls-files -z（repo=E:\论文2修改\IRRA\IRRA-main-mraf）; 对每个候选执行 git show HEAD:<path>，并以 UTF-8 读取工作树后比较 CRLF/LF 归一化结果 | WARN |
| 仅 EOL | IRRA-main/run_synthcap_yunwu_smoke.sh | git ls-files -z（repo=E:\论文2修改\IRRA\IRRA-main-mraf）; 对每个候选执行 git show HEAD:<path>，并以 UTF-8 读取工作树后比较 CRLF/LF 归一化结果 | WARN |
| 仅 EOL | IRRA-main/run_uspc_v1_ablation_60ep.sh | git ls-files -z（repo=E:\论文2修改\IRRA\IRRA-main-mraf）; 对每个候选执行 git show HEAD:<path>，并以 UTF-8 读取工作树后比较 CRLF/LF 归一化结果 | WARN |
| 仅 EOL | IRRA-main/run_uspc_v1_smoke_5ep.sh | git ls-files -z（repo=E:\论文2修改\IRRA\IRRA-main-mraf）; 对每个候选执行 git show HEAD:<path>，并以 UTF-8 读取工作树后比较 CRLF/LF 归一化结果 | WARN |
| 仅 EOL | IRRA-main/run_uspc_v2a_attr_smoke_2ep.sh | git ls-files -z（repo=E:\论文2修改\IRRA\IRRA-main-mraf）; 对每个候选执行 git show HEAD:<path>，并以 UTF-8 读取工作树后比较 CRLF/LF 归一化结果 | WARN |
| 仅 EOL | IRRA-main/run_uspc_v2b0_slot_local_20ep_grid.sh | git ls-files -z（repo=E:\论文2修改\IRRA\IRRA-main-mraf）; 对每个候选执行 git show HEAD:<path>，并以 UTF-8 读取工作树后比较 CRLF/LF 归一化结果 | WARN |
| 仅 EOL | IRRA-main/run_uspc_v2b0_slot_local_smoke_2ep.sh | git ls-files -z（repo=E:\论文2修改\IRRA\IRRA-main-mraf）; 对每个候选执行 git show HEAD:<path>，并以 UTF-8 读取工作树后比较 CRLF/LF 归一化结果 | WARN |
| 仅 EOL | IRRA-main/scripts/revision/README_AutoDL.md | git ls-files -z（repo=E:\论文2修改\IRRA\IRRA-main-mraf）; 对每个候选执行 git show HEAD:<path>，并以 UTF-8 读取工作树后比较 CRLF/LF 归一化结果 | WARN |
| 仅 EOL | IRRA-main/scripts/revision/run_new_degradation_matrix.sh | git ls-files -z（repo=E:\论文2修改\IRRA\IRRA-main-mraf）; 对每个候选执行 git show HEAD:<path>，并以 UTF-8 读取工作树后比较 CRLF/LF 归一化结果 | WARN |
| 仅 EOL | IRRA-main/scripts/revision/run_old_degradation_regression.sh | git ls-files -z（repo=E:\论文2修改\IRRA\IRRA-main-mraf）; 对每个候选执行 git show HEAD:<path>，并以 UTF-8 读取工作树后比较 CRLF/LF 归一化结果 | WARN |
| 仅 EOL | IRRA-main/solver/__init__.py | git ls-files -z（repo=E:\论文2修改\IRRA\IRRA-main-mraf）; 对每个候选执行 git show HEAD:<path>，并以 UTF-8 读取工作树后比较 CRLF/LF 归一化结果 | WARN |
| 仅 EOL | IRRA-main/solver/build.py | git ls-files -z（repo=E:\论文2修改\IRRA\IRRA-main-mraf）; 对每个候选执行 git show HEAD:<path>，并以 UTF-8 读取工作树后比较 CRLF/LF 归一化结果 | WARN |
| 仅 EOL | IRRA-main/solver/lr_scheduler.py | git ls-files -z（repo=E:\论文2修改\IRRA\IRRA-main-mraf）; 对每个候选执行 git show HEAD:<path>，并以 UTF-8 读取工作树后比较 CRLF/LF 归一化结果 | WARN |
| 仅 EOL | IRRA-main/sweep_lambda_unc_5ep.bat | git ls-files -z（repo=E:\论文2修改\IRRA\IRRA-main-mraf）; 对每个候选执行 git show HEAD:<path>，并以 UTF-8 读取工作树后比较 CRLF/LF 归一化结果 | WARN |
| 仅 EOL | IRRA-main/test.py | git ls-files -z（repo=E:\论文2修改\IRRA\IRRA-main-mraf）; 对每个候选执行 git show HEAD:<path>，并以 UTF-8 读取工作树后比较 CRLF/LF 归一化结果 | WARN |
| 仅 EOL | IRRA-main/tests/__init__.py | git ls-files -z（repo=E:\论文2修改\IRRA\IRRA-main-mraf）; 对每个候选执行 git show HEAD:<path>，并以 UTF-8 读取工作树后比较 CRLF/LF 归一化结果 | WARN |
| 仅 EOL | IRRA-main/tests/test_evaluation_degradation.py | git ls-files -z（repo=E:\论文2修改\IRRA\IRRA-main-mraf）; 对每个候选执行 git show HEAD:<path>，并以 UTF-8 读取工作树后比较 CRLF/LF 归一化结果 | WARN |
| 仅 EOL | IRRA-main/tests/test_revision_analysis_tools.py | git ls-files -z（repo=E:\论文2修改\IRRA\IRRA-main-mraf）; 对每个候选执行 git show HEAD:<path>，并以 UTF-8 读取工作树后比较 CRLF/LF 归一化结果 | WARN |
| 仅 EOL | IRRA-main/tools/analyze_qalr_v3_candidate_oracle.py | git ls-files -z（repo=E:\论文2修改\IRRA\IRRA-main-mraf）; 对每个候选执行 git show HEAD:<path>，并以 UTF-8 读取工作树后比较 CRLF/LF 归一化结果 | WARN |
| 仅 EOL | IRRA-main/tools/build_large_synth_bank_v2.py | git ls-files -z（repo=E:\论文2修改\IRRA\IRRA-main-mraf）; 对每个候选执行 git show HEAD:<path>，并以 UTF-8 读取工作树后比较 CRLF/LF 归一化结果 | WARN |
| 仅 EOL | IRRA-main/tools/build_large_synth_bank_v3.py | git ls-files -z（repo=E:\论文2修改\IRRA\IRRA-main-mraf）; 对每个候选执行 git show HEAD:<path>，并以 UTF-8 读取工作树后比较 CRLF/LF 归一化结果 | WARN |
| 仅 EOL | IRRA-main/tools/build_large_synth_bank_v4.py | git ls-files -z（repo=E:\论文2修改\IRRA\IRRA-main-mraf）; 对每个候选执行 git show HEAD:<path>，并以 UTF-8 读取工作树后比较 CRLF/LF 归一化结果 | WARN |
| 仅 EOL | IRRA-main/tools/build_large_synth_bank_v5_llm.py | git ls-files -z（repo=E:\论文2修改\IRRA\IRRA-main-mraf）; 对每个候选执行 git show HEAD:<path>，并以 UTF-8 读取工作树后比较 CRLF/LF 归一化结果 | WARN |
| 仅 EOL | IRRA-main/tools/build_mga_pseudo_from_captions.py | git ls-files -z（repo=E:\论文2修改\IRRA\IRRA-main-mraf）; 对每个候选执行 git show HEAD:<path>，并以 UTF-8 读取工作树后比较 CRLF/LF 归一化结果 | WARN |
| 仅 EOL | IRRA-main/tools/build_offline_hardneg_index.py | git ls-files -z（repo=E:\论文2修改\IRRA\IRRA-main-mraf）; 对每个候选执行 git show HEAD:<path>，并以 UTF-8 读取工作树后比较 CRLF/LF 归一化结果 | WARN |
| 仅 EOL | IRRA-main/tools/build_offline_prototype_bank.py | git ls-files -z（repo=E:\论文2修改\IRRA\IRRA-main-mraf）; 对每个候选执行 git show HEAD:<path>，并以 UTF-8 读取工作树后比较 CRLF/LF 归一化结果 | WARN |
| 仅 EOL | IRRA-main/tools/build_qalr_v4_gate_dataset.py | git ls-files -z（repo=E:\论文2修改\IRRA\IRRA-main-mraf）; 对每个候选执行 git show HEAD:<path>，并以 UTF-8 读取工作树后比较 CRLF/LF 归一化结果 | WARN |
| 仅 EOL | IRRA-main/tools/build_synth_bank_smoketest.py | git ls-files -z（repo=E:\论文2修改\IRRA\IRRA-main-mraf）; 对每个候选执行 git show HEAD:<path>，并以 UTF-8 读取工作树后比较 CRLF/LF 归一化结果 | WARN |
| 仅 EOL | IRRA-main/tools/build_synth_bank_v6_controlled.py | git ls-files -z（repo=E:\论文2修改\IRRA\IRRA-main-mraf）; 对每个候选执行 git show HEAD:<path>，并以 UTF-8 读取工作树后比较 CRLF/LF 归一化结果 | WARN |
| 仅 EOL | IRRA-main/tools/check_synth_bank_v6_controlled.py | git ls-files -z（repo=E:\论文2修改\IRRA\IRRA-main-mraf）; 对每个候选执行 git show HEAD:<path>，并以 UTF-8 读取工作树后比较 CRLF/LF 归一化结果 | WARN |
| 仅 EOL | IRRA-main/tools/check_synth_caption_bank.py | git ls-files -z（repo=E:\论文2修改\IRRA\IRRA-main-mraf）; 对每个候选执行 git show HEAD:<path>，并以 UTF-8 读取工作树后比较 CRLF/LF 归一化结果 | WARN |
| 仅 EOL | IRRA-main/tools/dump_rgbt_pedes_test_metadata.py | git ls-files -z（repo=E:\论文2修改\IRRA\IRRA-main-mraf）; 对每个候选执行 git show HEAD:<path>，并以 UTF-8 读取工作树后比较 CRLF/LF 归一化结果 | WARN |
| 仅 EOL | IRRA-main/tools/ensemble_eval_scores.py | git ls-files -z（repo=E:\论文2修改\IRRA\IRRA-main-mraf）; 对每个候选执行 git show HEAD:<path>，并以 UTF-8 读取工作树后比较 CRLF/LF 归一化结果 | WARN |
| 仅 EOL | IRRA-main/tools/revision/__init__.py | git ls-files -z（repo=E:\论文2修改\IRRA\IRRA-main-mraf）; 对每个候选执行 git show HEAD:<path>，并以 UTF-8 读取工作树后比较 CRLF/LF 归一化结果 | WARN |
| 仅 EOL | IRRA-main/tools/revision/color_group_evaluation.py | git ls-files -z（repo=E:\论文2修改\IRRA\IRRA-main-mraf）; 对每个候选执行 git show HEAD:<path>，并以 UTF-8 读取工作树后比较 CRLF/LF 归一化结果 | WARN |
| 仅 EOL | IRRA-main/tools/revision/color_terms_en.txt | git ls-files -z（repo=E:\论文2修改\IRRA\IRRA-main-mraf）; 对每个候选执行 git show HEAD:<path>，并以 UTF-8 读取工作树后比较 CRLF/LF 归一化结果 | WARN |
| 仅 EOL | IRRA-main/tools/revision/compare_score_files.py | git ls-files -z（repo=E:\论文2修改\IRRA\IRRA-main-mraf）; 对每个候选执行 git show HEAD:<path>，并以 UTF-8 读取工作树后比较 CRLF/LF 归一化结果 | WARN |
| 仅 EOL | IRRA-main/tools/revision/ensemble_scores.py | git ls-files -z（repo=E:\论文2修改\IRRA\IRRA-main-mraf）; 对每个候选执行 git show HEAD:<path>，并以 UTF-8 读取工作树后比较 CRLF/LF 归一化结果 | WARN |
| 仅 EOL | IRRA-main/tools/revision/feature_norm_analysis.py | git ls-files -z（repo=E:\论文2修改\IRRA\IRRA-main-mraf）; 对每个候选执行 git show HEAD:<path>，并以 UTF-8 读取工作树后比较 CRLF/LF 归一化结果 | WARN |
| 仅 EOL | IRRA-main/tools/revision/intensity_curve_summary.py | git ls-files -z（repo=E:\论文2修改\IRRA\IRRA-main-mraf）; 对每个候选执行 git show HEAD:<path>，并以 UTF-8 读取工作树后比较 CRLF/LF 归一化结果 | WARN |
| 仅 EOL | IRRA-main/tools/revision/noise_repeat_summary.py | git ls-files -z（repo=E:\论文2修改\IRRA\IRRA-main-mraf）; 对每个候选执行 git show HEAD:<path>，并以 UTF-8 读取工作树后比较 CRLF/LF 归一化结果 | WARN |
| 仅 EOL | IRRA-main/tools/revision/paired_seed_statistics.py | git ls-files -z（repo=E:\论文2修改\IRRA\IRRA-main-mraf）; 对每个候选执行 git show HEAD:<path>，并以 UTF-8 读取工作树后比较 CRLF/LF 归一化结果 | WARN |
| 仅 EOL | IRRA-main/tools/revision/score_io.py | git ls-files -z（repo=E:\论文2修改\IRRA\IRRA-main-mraf）; 对每个候选执行 git show HEAD:<path>，并以 UTF-8 读取工作树后比较 CRLF/LF 归一化结果 | WARN |
| 仅 EOL | IRRA-main/tools/revision/summarize_robustness.py | git ls-files -z（repo=E:\论文2修改\IRRA\IRRA-main-mraf）; 对每个候选执行 git show HEAD:<path>，并以 UTF-8 读取工作树后比较 CRLF/LF 归一化结果 | WARN |
| 仅 EOL | IRRA-main/tools/run_qalr_v2_safe_grid.sh | git ls-files -z（repo=E:\论文2修改\IRRA\IRRA-main-mraf）; 对每个候选执行 git show HEAD:<path>，并以 UTF-8 读取工作树后比较 CRLF/LF 归一化结果 | WARN |
| 仅 EOL | IRRA-main/tools/smoke_mga_dataloader.py | git ls-files -z（repo=E:\论文2修改\IRRA\IRRA-main-mraf）; 对每个候选执行 git show HEAD:<path>，并以 UTF-8 读取工作树后比较 CRLF/LF 归一化结果 | WARN |
| 仅 EOL | IRRA-main/tools/train_qalr_v4_gate.py | git ls-files -z（repo=E:\论文2修改\IRRA\IRRA-main-mraf）; 对每个候选执行 git show HEAD:<path>，并以 UTF-8 读取工作树后比较 CRLF/LF 归一化结果 | WARN |
| 仅 EOL | IRRA-main/tools/validate_mga_pseudo_json.py | git ls-files -z（repo=E:\论文2修改\IRRA\IRRA-main-mraf）; 对每个候选执行 git show HEAD:<path>，并以 UTF-8 读取工作树后比较 CRLF/LF 归一化结果 | WARN |
| 仅 EOL | IRRA-main/tools/weight_split_eval.py | git ls-files -z（repo=E:\论文2修改\IRRA\IRRA-main-mraf）; 对每个候选执行 git show HEAD:<path>，并以 UTF-8 读取工作树后比较 CRLF/LF 归一化结果 | WARN |
| 仅 EOL | IRRA-main/train.py | git ls-files -z（repo=E:\论文2修改\IRRA\IRRA-main-mraf）; 对每个候选执行 git show HEAD:<path>，并以 UTF-8 读取工作树后比较 CRLF/LF 归一化结果 | WARN |
| 仅 EOL | IRRA-main/utils/__init__.py | git ls-files -z（repo=E:\论文2修改\IRRA\IRRA-main-mraf）; 对每个候选执行 git show HEAD:<path>，并以 UTF-8 读取工作树后比较 CRLF/LF 归一化结果 | WARN |
| 仅 EOL | IRRA-main/utils/attribute_parser.py | git ls-files -z（repo=E:\论文2修改\IRRA\IRRA-main-mraf）; 对每个候选执行 git show HEAD:<path>，并以 UTF-8 读取工作树后比较 CRLF/LF 归一化结果 | WARN |
| 仅 EOL | IRRA-main/utils/checkpoint.py | git ls-files -z（repo=E:\论文2修改\IRRA\IRRA-main-mraf）; 对每个候选执行 git show HEAD:<path>，并以 UTF-8 读取工作树后比较 CRLF/LF 归一化结果 | WARN |
| 仅 EOL | IRRA-main/utils/comm.py | git ls-files -z（repo=E:\论文2修改\IRRA\IRRA-main-mraf）; 对每个候选执行 git show HEAD:<path>，并以 UTF-8 读取工作树后比较 CRLF/LF 归一化结果 | WARN |
| 仅 EOL | IRRA-main/utils/eval_degradation.py | git ls-files -z（repo=E:\论文2修改\IRRA\IRRA-main-mraf）; 对每个候选执行 git show HEAD:<path>，并以 UTF-8 读取工作树后比较 CRLF/LF 归一化结果 | WARN |
| 仅 EOL | IRRA-main/utils/iotools.py | git ls-files -z（repo=E:\论文2修改\IRRA\IRRA-main-mraf）; 对每个候选执行 git show HEAD:<path>，并以 UTF-8 读取工作树后比较 CRLF/LF 归一化结果 | WARN |
| 仅 EOL | IRRA-main/utils/logger.py | git ls-files -z（repo=E:\论文2修改\IRRA\IRRA-main-mraf）; 对每个候选执行 git show HEAD:<path>，并以 UTF-8 读取工作树后比较 CRLF/LF 归一化结果 | WARN |
| 仅 EOL | IRRA-main/utils/mdrc.py | git ls-files -z（repo=E:\论文2修改\IRRA\IRRA-main-mraf）; 对每个候选执行 git show HEAD:<path>，并以 UTF-8 读取工作树后比较 CRLF/LF 归一化结果 | WARN |
| 仅 EOL | IRRA-main/utils/meter.py | git ls-files -z（repo=E:\论文2修改\IRRA\IRRA-main-mraf）; 对每个候选执行 git show HEAD:<path>，并以 UTF-8 读取工作树后比较 CRLF/LF 归一化结果 | WARN |
| 仅 EOL | IRRA-main/utils/metrics.py | git ls-files -z（repo=E:\论文2修改\IRRA\IRRA-main-mraf）; 对每个候选执行 git show HEAD:<path>，并以 UTF-8 读取工作树后比较 CRLF/LF 归一化结果 | WARN |
| 仅 EOL | IRRA-main/utils/mtp_text.py | git ls-files -z（repo=E:\论文2修改\IRRA\IRRA-main-mraf）; 对每个候选执行 git show HEAD:<path>，并以 UTF-8 读取工作树后比较 CRLF/LF 归一化结果 | WARN |
| 仅 EOL | IRRA-main/utils/qalr_v4_gate.py | git ls-files -z（repo=E:\论文2修改\IRRA\IRRA-main-mraf）; 对每个候选执行 git show HEAD:<path>，并以 UTF-8 读取工作树后比较 CRLF/LF 归一化结果 | WARN |
| 仅 EOL | IRRA-main/utils/query_adaptive_rerank.py | git ls-files -z（repo=E:\论文2修改\IRRA\IRRA-main-mraf）; 对每个候选执行 git show HEAD:<path>，并以 UTF-8 读取工作树后比较 CRLF/LF 归一化结果 | WARN |
| 仅 EOL | IRRA-main/utils/simple_tokenizer.py | git ls-files -z（repo=E:\论文2修改\IRRA\IRRA-main-mraf）; 对每个候选执行 git show HEAD:<path>，并以 UTF-8 读取工作树后比较 CRLF/LF 归一化结果 | WARN |
| 仅 EOL | IRRA-main/utils/transport_rerank.py | git ls-files -z（repo=E:\论文2修改\IRRA\IRRA-main-mraf）; 对每个候选执行 git show HEAD:<path>，并以 UTF-8 读取工作树后比较 CRLF/LF 归一化结果 | WARN |
| 仅 EOL | IRRA-main/utils/uspc_text_views.py | git ls-files -z（repo=E:\论文2修改\IRRA\IRRA-main-mraf）; 对每个候选执行 git show HEAD:<path>，并以 UTF-8 读取工作树后比较 CRLF/LF 归一化结果 | WARN |
| 仅 EOL | IRRA-main/visualize.py | git ls-files -z（repo=E:\论文2修改\IRRA\IRRA-main-mraf）; 对每个候选执行 git show HEAD:<path>，并以 UTF-8 读取工作树后比较 CRLF/LF 归一化结果 | WARN |
| 其它语义差异 | .gitignore | git ls-files -z（repo=E:\论文2修改\IRRA\IRRA-main-mraf）; 对每个候选执行 git show HEAD:<path>，并以 UTF-8 读取工作树后比较 CRLF/LF 归一化结果 | WARN |
