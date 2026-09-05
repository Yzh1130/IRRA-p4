# 仓库状态审计报告

本报告仅记录 Git 只读命令在采集时返回的事实；状态含义不替代人工裁定。

## 一、基本状态

| 检查项 | 值 | 状态 | 来源 | 备注 |
| --- | --- | --- | --- | --- |
| 采集时间 | 2026-09-05T17:16:04+08:00 | OK | Python datetime.now().astimezone() | 本地时区 ISO 8601 时间 |
| 命令行仓库路径 | E:\论文2修改\IRRA\IRRA-main-mraf | OK | 命令行参数 --repo |  |
| Git 顶层目录 | E:\论文2修改\IRRA\IRRA-main-mraf | OK | git rev-parse --show-toplevel（repo=E:\论文2修改\IRRA\IRRA-main-mraf） |  |
| 当前 branch | stage0/audit-tooling | OK | git branch --show-current（repo=E:\论文2修改\IRRA\IRRA-main-mraf） |  |
| HEAD commit | d07171be5c2e12f9439c45568c9047b52ef8596d | OK | git rev-parse HEAD（repo=E:\论文2修改\IRRA\IRRA-main-mraf） |  |
| 未提交改动数量 | 13 | WARN | git status --porcelain=v1 -z --untracked-files=all（repo=E:\论文2修改\IRRA\IRRA-main-mraf） | 数量按 porcelain -z 记录计数 |

## 二、Remote 列表

| 名称 | URL | 用途 | 状态 | 来源 | 备注 |
| --- | --- | --- | --- | --- | --- |
| origin | https://gitee.com/Yanzh1130/IRRA.git | fetch | OK | git remote -v（repo=E:\论文2修改\IRRA\IRRA-main-mraf） |  |
| origin | https://gitee.com/Yanzh1130/IRRA.git | push | OK | git remote -v（repo=E:\论文2修改\IRRA\IRRA-main-mraf） |  |

## 三、本地与远程分支

| 范围 | 分支 | 状态 | 来源 | 备注 |
| --- | --- | --- | --- | --- |
| 本地 | backup_cd2fa20e_clean | OK | git branch --list --format=%(refname:short)（repo=E:\论文2修改\IRRA\IRRA-main-mraf） |  |
| 本地 | backup_new_local_feature | OK | git branch --list --format=%(refname:short)（repo=E:\论文2修改\IRRA\IRRA-main-mraf） |  |
| 本地 | diagnostics/mraf-mechanism-v1 | OK | git branch --list --format=%(refname:short)（repo=E:\论文2修改\IRRA\IRRA-main-mraf） |  |
| 本地 | experiment/mraf-gate-amplitude-rescue-v3 | OK | git branch --list --format=%(refname:short)（repo=E:\论文2修改\IRRA\IRRA-main-mraf） |  |
| 本地 | experiment/mraf-gate-rescue-v2 | OK | git branch --list --format=%(refname:short)（repo=E:\论文2修改\IRRA\IRRA-main-mraf） |  |
| 本地 | experiment/mraf-ir-sensitivity-rescue-v4 | OK | git branch --list --format=%(refname:short)（repo=E:\论文2修改\IRRA\IRRA-main-mraf） |  |
| 本地 | experiment/mraf-norm-equalization | OK | git branch --list --format=%(refname:short)（repo=E:\论文2修改\IRRA\IRRA-main-mraf） |  |
| 本地 | feature/mraf-reliability-fusion | OK | git branch --list --format=%(refname:short)（repo=E:\论文2修改\IRRA\IRRA-main-mraf） |  |
| 本地 | master | OK | git branch --list --format=%(refname:short)（repo=E:\论文2修改\IRRA\IRRA-main-mraf） |  |
| 本地 | mdrc_from_57711 | OK | git branch --list --format=%(refname:short)（repo=E:\论文2修改\IRRA\IRRA-main-mraf） |  |
| 本地 | mdrc_v1_from_cd2fa20e | OK | git branch --list --format=%(refname:short)（repo=E:\论文2修改\IRRA\IRRA-main-mraf） |  |
| 本地 | paper_revision_20260710 | OK | git branch --list --format=%(refname:short)（repo=E:\论文2修改\IRRA\IRRA-main-mraf） |  |
| 本地 | stage0/audit-tooling | OK | git branch --list --format=%(refname:short)（repo=E:\论文2修改\IRRA\IRRA-main-mraf） |  |
| 远程 | origin | OK | git branch --remotes --format=%(refname:short)（repo=E:\论文2修改\IRRA\IRRA-main-mraf） |  |
| 远程 | origin/diagnostics/mraf-mechanism-v1 | OK | git branch --remotes --format=%(refname:short)（repo=E:\论文2修改\IRRA\IRRA-main-mraf） |  |
| 远程 | origin/experiment/mraf-gate-amplitude-rescue-v3 | OK | git branch --remotes --format=%(refname:short)（repo=E:\论文2修改\IRRA\IRRA-main-mraf） |  |
| 远程 | origin/experiment/mraf-gate-rescue-v2 | OK | git branch --remotes --format=%(refname:short)（repo=E:\论文2修改\IRRA\IRRA-main-mraf） |  |
| 远程 | origin/experiment/mraf-ir-sensitivity-rescue-v4 | OK | git branch --remotes --format=%(refname:short)（repo=E:\论文2修改\IRRA\IRRA-main-mraf） |  |
| 远程 | origin/feature/mraf-reliability-fusion | OK | git branch --remotes --format=%(refname:short)（repo=E:\论文2修改\IRRA\IRRA-main-mraf） |  |
| 远程 | origin/master | OK | git branch --remotes --format=%(refname:short)（repo=E:\论文2修改\IRRA\IRRA-main-mraf） |  |
| 远程 | origin/mdrc_v1_from_cd2fa20e | OK | git branch --remotes --format=%(refname:short)（repo=E:\论文2修改\IRRA\IRRA-main-mraf） |  |
| 远程 | origin/paper_revision_20260710 | OK | git branch --remotes --format=%(refname:short)（repo=E:\论文2修改\IRRA\IRRA-main-mraf） |  |

## 四、Tags

| Tag | 状态 | 来源 | 备注 |
| --- | --- | --- | --- |
| 未找到 | WARN | git tag --list（repo=E:\论文2修改\IRRA\IRRA-main-mraf） | 仓库没有 tag |

## 五、Worktree 列表

| 路径 | HEAD | 分支 | Locked | Prunable | 状态 | 来源 | 备注 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| E:/论文2修改/IRRA | 5f787e67ca32404fee0256b78829456aed5a9484 | paper_revision_20260710 | 否 | 否 | OK | git worktree list --porcelain（repo=E:\论文2修改\IRRA\IRRA-main-mraf） |  |
| E:/论文2修改/IRRA/IRRA-main-mraf | d07171be5c2e12f9439c45568c9047b52ef8596d | stage0/audit-tooling | 否 | 否 | OK | git worktree list --porcelain（repo=E:\论文2修改\IRRA\IRRA-main-mraf） |  |

## 六、未提交改动与逐文件 diff --stat

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

## 七、最近 50 条 Commit

| Hash | 时间 | 作者 | 标题 | 状态 | 来源 | 备注 |
| --- | --- | --- | --- | --- | --- | --- |
| d07171be5c2e12f9439c45568c9047b52ef8596d | 2026-07-28T01:00:20+08:00 | Yzh | feat: add MRAF norm equalization ablations | OK | git log -n 50 --date=iso-strict --no-decorate --pretty=format:%H%x1f%aI%x1f%an%x1f%s%x1e（repo=E:\论文2修改\IRRA\IRRA-main-mraf） |  |
| 6b469e2b3e18441aaed755308f4ca4130a12585f | 2026-07-27T16:54:16+08:00 | Yzh | feat: add MRAF IR sensitivity rescue and multiseed pipeline | OK | git log -n 50 --date=iso-strict --no-decorate --pretty=format:%H%x1f%aI%x1f%an%x1f%s%x1e（repo=E:\论文2修改\IRRA\IRRA-main-mraf） |  |
| f6629620ea6750f94253f79a72d3a8ff4a94e378 | 2026-07-27T13:18:48+08:00 | Yzh | fix: resume MRAF amplitude pipeline after calibration | OK | git log -n 50 --date=iso-strict --no-decorate --pretty=format:%H%x1f%aI%x1f%an%x1f%s%x1e（repo=E:\论文2修改\IRRA\IRRA-main-mraf） |  |
| ed217ac5875c31a2d818fb643554b8fd8dcbc732 | 2026-07-27T11:28:45+08:00 | Yzh | fix: make MRAF resume tests portable on Linux | OK | git log -n 50 --date=iso-strict --no-decorate --pretty=format:%H%x1f%aI%x1f%an%x1f%s%x1e（repo=E:\论文2修改\IRRA\IRRA-main-mraf） |  |
| 23d7be826252f1fd3c0a0364462bf7e22b995dd9 | 2026-07-27T11:09:06+08:00 | Yzh | fix: resume MRAF amplitude screen from legacy outputs | OK | git log -n 50 --date=iso-strict --no-decorate --pretty=format:%H%x1f%aI%x1f%an%x1f%s%x1e（repo=E:\论文2修改\IRRA\IRRA-main-mraf） |  |
| c667946b2c51740bfa7892f5a2daee73d1279b44 | 2026-07-26T18:25:02+08:00 | Yzh | feat: add MRAF gate amplitude rescue pipeline | OK | git log -n 50 --date=iso-strict --no-decorate --pretty=format:%H%x1f%aI%x1f%an%x1f%s%x1e（repo=E:\论文2修改\IRRA\IRRA-main-mraf） |  |
| da8c836e9b2d96605d5672dea2af9dc166e60ea2 | 2026-07-25T22:39:27+08:00 | Yzh | fix: support historical tuple-tagged MRAF configs | OK | git log -n 50 --date=iso-strict --no-decorate --pretty=format:%H%x1f%aI%x1f%an%x1f%s%x1e（repo=E:\论文2修改\IRRA\IRRA-main-mraf） |  |
| 67f138a1c56956608e752a88e03d118efac7b52c | 2026-07-25T18:45:42+08:00 | Yzh | feat: add automated MRAF gate rescue pipeline | OK | git log -n 50 --date=iso-strict --no-decorate --pretty=format:%H%x1f%aI%x1f%an%x1f%s%x1e（repo=E:\论文2修改\IRRA\IRRA-main-mraf） |  |
| 1a5f5a752e8408d9bae954219858845036807b13 | 2026-07-25T15:54:10+08:00 | Yzh | docs: update MRAF diagnostic memory guidance | OK | git log -n 50 --date=iso-strict --no-decorate --pretty=format:%H%x1f%aI%x1f%an%x1f%s%x1e（repo=E:\论文2修改\IRRA\IRRA-main-mraf） |  |
| 7d37330a4edac269a815bb94341952e7a5abd7ad | 2026-07-25T15:39:21+08:00 | Yzh | fix: bound MRAF diagnostic GPU memory | OK | git log -n 50 --date=iso-strict --no-decorate --pretty=format:%H%x1f%aI%x1f%an%x1f%s%x1e（repo=E:\论文2修改\IRRA\IRRA-main-mraf） |  |
| c116a50220903fcbe3c60cf3d0cfc832b7b6469f | 2026-07-25T14:41:29+08:00 | Yzh | test: add MRAF mechanism diagnostics | OK | git log -n 50 --date=iso-strict --no-decorate --pretty=format:%H%x1f%aI%x1f%an%x1f%s%x1e（repo=E:\论文2修改\IRRA\IRRA-main-mraf） |  |
| 2a647ab50da6440002d3968a0e1e4cbbf243bf02 | 2026-07-24T16:03:20+08:00 | Yzh | feat: add reliability-adaptive RGB-T fusion for MDRC | OK | git log -n 50 --date=iso-strict --no-decorate --pretty=format:%H%x1f%aI%x1f%an%x1f%s%x1e（repo=E:\论文2修改\IRRA\IRRA-main-mraf） |  |
| 5f787e67ca32404fee0256b78829456aed5a9484 | 2026-07-10T19:53:18+08:00 | Yzh | add evaluation degradations and robustness analysis tools | OK | git log -n 50 --date=iso-strict --no-decorate --pretty=format:%H%x1f%aI%x1f%an%x1f%s%x1e（repo=E:\论文2修改\IRRA\IRRA-main-mraf） |  |
| 87610d503a8a30962249a2b301d6787f5d39b7af | 2026-07-08T15:00:57+08:00 | Yzh | add ablation degradation augmentation training mode | OK | git log -n 50 --date=iso-strict --no-decorate --pretty=format:%H%x1f%aI%x1f%an%x1f%s%x1e（repo=E:\论文2修改\IRRA\IRRA-main-mraf） |  |
| eba387f48fb733d51d8e9a6408744ff69464221f | 2026-07-08T14:52:03+08:00 | Yzh | add split-half weighted fusion eval tool | OK | git log -n 50 --date=iso-strict --no-decorate --pretty=format:%H%x1f%aI%x1f%an%x1f%s%x1e（repo=E:\论文2修改\IRRA\IRRA-main-mraf） |  |
| 9db44fb2fe218981ddda3612adb6d3dd0490f077 | 2026-07-05T00:19:35+08:00 | Yzh | add rgbt pedes test metadata dump tool | OK | git log -n 50 --date=iso-strict --no-decorate --pretty=format:%H%x1f%aI%x1f%an%x1f%s%x1e（repo=E:\论文2修改\IRRA\IRRA-main-mraf） |  |
| 6cd6bddaee180c7a5f25ae32deaafeb019a3e883 | 2026-07-03T22:54:44+08:00 | Yzh | add evaluation-only degradation test entry | OK | git log -n 50 --date=iso-strict --no-decorate --pretty=format:%H%x1f%aI%x1f%an%x1f%s%x1e（repo=E:\论文2修改\IRRA\IRRA-main-mraf） |  |
| a37d4e7a0237ede5044106910a8050b7d5e4a90c | 2026-07-03T17:19:52+08:00 | Yzh | add MDRC loss scale | OK | git log -n 50 --date=iso-strict --no-decorate --pretty=format:%H%x1f%aI%x1f%an%x1f%s%x1e（repo=E:\论文2修改\IRRA\IRRA-main-mraf） |  |
| 96239ab8dcda8546c5e1fa271f2151f5fe6918bf | 2026-07-03T17:02:56+08:00 | Yzh | fix MDRC loss integration and debug logging | OK | git log -n 50 --date=iso-strict --no-decorate --pretty=format:%H%x1f%aI%x1f%an%x1f%s%x1e（repo=E:\论文2修改\IRRA\IRRA-main-mraf） |  |
| d3f59ed0db5621509d9f602c625bf28f13c373cb | 2026-07-03T16:08:42+08:00 | Yzh | fix MDRC degraded feature-only forward | OK | git log -n 50 --date=iso-strict --no-decorate --pretty=format:%H%x1f%aI%x1f%an%x1f%s%x1e（repo=E:\论文2修改\IRRA\IRRA-main-mraf） |  |
| 890bd000fa72028ec0c524e37235fcd1bfbe1002 | 2026-07-03T15:24:16+08:00 | Yzh | add MDRC v1 clean-degraded consistency | OK | git log -n 50 --date=iso-strict --no-decorate --pretty=format:%H%x1f%aI%x1f%an%x1f%s%x1e（repo=E:\论文2修改\IRRA\IRRA-main-mraf） |  |
| cd2fa20e073e2d8b149ea321e4700cb012c50187 | 2026-06-28T16:00:06+08:00 | Yzh | update gitignore | OK | git log -n 50 --date=iso-strict --no-decorate --pretty=format:%H%x1f%aI%x1f%an%x1f%s%x1e（repo=E:\论文2修改\IRRA\IRRA-main-mraf） |  |
| 5c1614b6cd8c8e63aa03e9a84abfd7865aed806c | 2026-06-09T21:10:24+08:00 | Yzh | 2026年6月9日21:10:14 | OK | git log -n 50 --date=iso-strict --no-decorate --pretty=format:%H%x1f%aI%x1f%an%x1f%s%x1e（repo=E:\论文2修改\IRRA\IRRA-main-mraf） |  |
| 175476f890dda7804a536b7818ff5578ad5038d7 | 2026-06-09T20:30:30+08:00 | Yzh | 2026年6月9日20:30:20 | OK | git log -n 50 --date=iso-strict --no-decorate --pretty=format:%H%x1f%aI%x1f%an%x1f%s%x1e（repo=E:\论文2修改\IRRA\IRRA-main-mraf） |  |
| ce542fcf6494e069c0d5b425486056d58694dde8 | 2026-06-05T22:14:58+08:00 | Yzh | 2026年6月5日22:14:47 | OK | git log -n 50 --date=iso-strict --no-decorate --pretty=format:%H%x1f%aI%x1f%an%x1f%s%x1e（repo=E:\论文2修改\IRRA\IRRA-main-mraf） |  |
| caaa8bbd9a7524173a1edec4e23157d309ffc288 | 2026-06-05T21:53:36+08:00 | Yzh | 2026年6月5日21:53:10 | OK | git log -n 50 --date=iso-strict --no-decorate --pretty=format:%H%x1f%aI%x1f%an%x1f%s%x1e（repo=E:\论文2修改\IRRA\IRRA-main-mraf） |  |
| b70eeb79ef16e60370d329e709bb525133dfa81f | 2026-06-03T23:02:42+08:00 | Yzh | 2026年6月3日23:01:56 | OK | git log -n 50 --date=iso-strict --no-decorate --pretty=format:%H%x1f%aI%x1f%an%x1f%s%x1e（repo=E:\论文2修改\IRRA\IRRA-main-mraf） |  |
| 7d793e9a5f1681019914d99ade498bf08c78abab | 2026-06-03T22:52:39+08:00 | Yzh | 2026年6月3日21:44:05 | OK | git log -n 50 --date=iso-strict --no-decorate --pretty=format:%H%x1f%aI%x1f%an%x1f%s%x1e（repo=E:\论文2修改\IRRA\IRRA-main-mraf） |  |
| df1c012358a2b7352e2549a8be33133f8783fa89 | 2026-06-02T21:45:44+08:00 | Yzh | 2026年6月2日21:45:11 | OK | git log -n 50 --date=iso-strict --no-decorate --pretty=format:%H%x1f%aI%x1f%an%x1f%s%x1e（repo=E:\论文2修改\IRRA\IRRA-main-mraf） |  |
| fdf7aebc1c6d2ba65b1fbc3acdeb147d8e07a282 | 2026-06-02T18:00:33+08:00 | Yzh | 2026年6月2日18:00:17 | OK | git log -n 50 --date=iso-strict --no-decorate --pretty=format:%H%x1f%aI%x1f%an%x1f%s%x1e（repo=E:\论文2修改\IRRA\IRRA-main-mraf） |  |
| ab6bcacd4b4d8b1bc60e84fc7dab07d2df594852 | 2026-06-02T00:28:00+08:00 | Yzh | 2026年6月2日00:27:47 | OK | git log -n 50 --date=iso-strict --no-decorate --pretty=format:%H%x1f%aI%x1f%an%x1f%s%x1e（repo=E:\论文2修改\IRRA\IRRA-main-mraf） |  |
| 48ac87578a16a971d0be2a7440b8be13e2be96a5 | 2026-06-01T20:37:49+08:00 | Yzh | 2026年6月1日20:37:00 | OK | git log -n 50 --date=iso-strict --no-decorate --pretty=format:%H%x1f%aI%x1f%an%x1f%s%x1e（repo=E:\论文2修改\IRRA\IRRA-main-mraf） |  |
| 8033165c56c548cbb7c8f4dd671255a8f9a6e6e2 | 2026-06-01T16:43:36+08:00 | Yzh | 2026年6月1日16:43:17 | OK | git log -n 50 --date=iso-strict --no-decorate --pretty=format:%H%x1f%aI%x1f%an%x1f%s%x1e（repo=E:\论文2修改\IRRA\IRRA-main-mraf） |  |
| 3123b87455daa56a7df29b0173b36bb7b8b191da | 2026-05-27T19:10:49+08:00 | Yzh | 2026年5月27日19:10:39 | OK | git log -n 50 --date=iso-strict --no-decorate --pretty=format:%H%x1f%aI%x1f%an%x1f%s%x1e（repo=E:\论文2修改\IRRA\IRRA-main-mraf） |  |
| c243681596e425e226db2a120784f121fcc86f90 | 2026-05-26T18:49:16+08:00 | Yzh | 2026年5月26日18:48:53 | OK | git log -n 50 --date=iso-strict --no-decorate --pretty=format:%H%x1f%aI%x1f%an%x1f%s%x1e（repo=E:\论文2修改\IRRA\IRRA-main-mraf） |  |
| cacdedb2e0222f50bb5f05cfe0b1ef47829c1aa7 | 2026-05-25T17:43:45+08:00 | Yzh | 2026年5月25日17:43:30 | OK | git log -n 50 --date=iso-strict --no-decorate --pretty=format:%H%x1f%aI%x1f%an%x1f%s%x1e（repo=E:\论文2修改\IRRA\IRRA-main-mraf） |  |
| a6d22512df4a7b7dd0a799058c8da98a7151c4d4 | 2026-05-25T12:53:24+08:00 | Yzh | 2026年5月25日12:53:14 | OK | git log -n 50 --date=iso-strict --no-decorate --pretty=format:%H%x1f%aI%x1f%an%x1f%s%x1e（repo=E:\论文2修改\IRRA\IRRA-main-mraf） |  |
| 31cac3a0c170414dd603fa295fc1c1d08a253495 | 2026-05-25T12:19:32+08:00 | Yzh | 2026年5月25日12:18:58 | OK | git log -n 50 --date=iso-strict --no-decorate --pretty=format:%H%x1f%aI%x1f%an%x1f%s%x1e（repo=E:\论文2修改\IRRA\IRRA-main-mraf） |  |
| db0e6098c9b2814c65349c4e539cfbe805839c94 | 2026-05-21T09:48:32+08:00 | Yzh | 2026年5月21日09:48:22 | OK | git log -n 50 --date=iso-strict --no-decorate --pretty=format:%H%x1f%aI%x1f%an%x1f%s%x1e（repo=E:\论文2修改\IRRA\IRRA-main-mraf） |  |
| 5ebffabb911c9edb548dc4808f5e41ce4f4dd3b9 | 2026-05-20T01:01:08Z | Yanzh1130 | Add USPC v2B0 slot local attribute loss | OK | git log -n 50 --date=iso-strict --no-decorate --pretty=format:%H%x1f%aI%x1f%an%x1f%s%x1e（repo=E:\论文2修改\IRRA\IRRA-main-mraf） |  |
| 1f8dd661b1683447c4f09ebcf5c976e5e7e47c83 | 2026-05-12T08:55:37+08:00 | Yzh | 2026年5月12日08:55:30 | OK | git log -n 50 --date=iso-strict --no-decorate --pretty=format:%H%x1f%aI%x1f%an%x1f%s%x1e（repo=E:\论文2修改\IRRA\IRRA-main-mraf） |  |
| f4247d7242dbcc9a24506cb78cd62bae443ff973 | 2026-05-12T07:29:36+08:00 | Yzh | 2026年5月12日07:29:28 | OK | git log -n 50 --date=iso-strict --no-decorate --pretty=format:%H%x1f%aI%x1f%an%x1f%s%x1e（repo=E:\论文2修改\IRRA\IRRA-main-mraf） |  |
| d2046304f771cda3cf8d54bfed4afd501ac5629c | 2026-05-11T18:25:11+08:00 | Yzh | 2026年5月11日18:25:04 | OK | git log -n 50 --date=iso-strict --no-decorate --pretty=format:%H%x1f%aI%x1f%an%x1f%s%x1e（repo=E:\论文2修改\IRRA\IRRA-main-mraf） |  |
| 7251b1d1bdc01f5ec474534c9ba90212b06d61d5 | 2026-05-10T13:49:04+08:00 | Yzh | 2026年5月10日13:48:54 | OK | git log -n 50 --date=iso-strict --no-decorate --pretty=format:%H%x1f%aI%x1f%an%x1f%s%x1e（repo=E:\论文2修改\IRRA\IRRA-main-mraf） |  |
| 0abff89996fe4d77bd3e19335686a091390523cc | 2026-05-10T13:17:34+08:00 | Yzh | 2026年5月10日13:17:33 | OK | git log -n 50 --date=iso-strict --no-decorate --pretty=format:%H%x1f%aI%x1f%an%x1f%s%x1e（repo=E:\论文2修改\IRRA\IRRA-main-mraf） |  |
| ab781bd099186f0148199bee091a3c9fcd16c04f | 2026-05-09T22:59:24+08:00 | Yzh | 2026年5月9日22:59:22 | OK | git log -n 50 --date=iso-strict --no-decorate --pretty=format:%H%x1f%aI%x1f%an%x1f%s%x1e（repo=E:\论文2修改\IRRA\IRRA-main-mraf） |  |
| 36df183439613e3a9e9db67ca3589c8936a13e2f | 2026-05-09T20:02:18+08:00 | Yzh | 2026年5月9日20:02:14 | OK | git log -n 50 --date=iso-strict --no-decorate --pretty=format:%H%x1f%aI%x1f%an%x1f%s%x1e（repo=E:\论文2修改\IRRA\IRRA-main-mraf） |  |
| 2a883e9b6bd2dc5cadd645ec5e9f3278b84d4809 | 2026-05-09T00:36:00+08:00 | Yzh | 202659_00:35:48 | OK | git log -n 50 --date=iso-strict --no-decorate --pretty=format:%H%x1f%aI%x1f%an%x1f%s%x1e（repo=E:\论文2修改\IRRA\IRRA-main-mraf） |  |
| 30d2c3d5e918465fea89c4911e521427afee090b | 2026-05-08T19:45:01+08:00 | Yzh | 20260508_19:44:52 | OK | git log -n 50 --date=iso-strict --no-decorate --pretty=format:%H%x1f%aI%x1f%an%x1f%s%x1e（repo=E:\论文2修改\IRRA\IRRA-main-mraf） |  |
| d11512ae2ce0dad1ddfe9048b8c11f7d0c2234db | 2026-05-08T16:50:11+08:00 | Yzh | 20260508_165001 | OK | git log -n 50 --date=iso-strict --no-decorate --pretty=format:%H%x1f%aI%x1f%an%x1f%s%x1e（repo=E:\论文2修改\IRRA\IRRA-main-mraf） |  |

## 八、tracked 文本内容与 EOL 核验

| 检查项 | 数量/值 | 状态 | 来源 | 备注 |
| --- | --- | --- | --- | --- |
| 检查的 tracked 文本文件 | 154 | WARN | git ls-files -z（repo=E:\论文2修改\IRRA\IRRA-main-mraf）; 对每个候选执行 git show HEAD:<path>，并以 UTF-8 读取工作树后比较 CRLF/LF 归一化结果 | 不读取 datasets/TextReID，不写 index，不修改工作树。 |
| 仅 EOL 不同 | 118 | WARN | git ls-files -z（repo=E:\论文2修改\IRRA\IRRA-main-mraf）; 对每个候选执行 git show HEAD:<path>，并以 UTF-8 读取工作树后比较 CRLF/LF 归一化结果 | IRRA-main/.idea/.gitignore<br>IRRA-main/.idea/IRRA-main.iml<br>IRRA-main/.idea/inspectionProfiles/profiles_settings.xml<br>IRRA-main/.idea/misc.xml<br>IRRA-main/.idea/modules.xml<br>IRRA-main/LICENSE<br>IRRA-main/README.md<br>IRRA-main/datasets/__init__.py<br>IRRA-main/datasets/bases.py<br>IRRA-main/datasets/build.py<br>IRRA-main/datasets/cuhkpedes.py<br>IRRA-main/datasets/icfgpedes.py<br>IRRA-main/datasets/mga_pseudo.py<br>IRRA-main/datasets/preprocessing.py<br>IRRA-main/datasets/rgbtpedes.py<br>IRRA-main/datasets/rstpreid.py<br>IRRA-main/datasets/sampler.py<br>IRRA-main/datasets/sampler_ddp.py<br>IRRA-main/docs/MGA_RDE_USAGE.md<br>IRRA-main/docs/revision/degradation_implementation_facts.md<br>IRRA-main/docs/revision/local_validation_report.md<br>IRRA-main/eval_local.py<br>IRRA-main/model/__init__.py<br>IRRA-main/model/clip_model.py<br>IRRA-main/model/objectives.py<br>IRRA-main/model/prototype_bank.py<br>IRRA-main/model/rgbt_moe_adapter.py<br>IRRA-main/model/text_guided_masking.py<br>IRRA-main/model/uncertainty.py<br>IRRA-main/pack/20260420_094400_synthv5_rollout/banks/synth_caption_bank_smoketest.json<br>IRRA-main/pack/20260420_094400_synthv5_rollout/vllm_pid.txt<br>IRRA-main/pack/20260420_094400_synthv5_rollout/vllm_port.txt<br>IRRA-main/pack/20260428_193242_hardneg_v2_sweep/run_hardneg_v2_all.sh<br>IRRA-main/pack/20260428_193242_hardneg_v2_sweep/summarize_hardneg_v2.py<br>IRRA-main/pack/20260429_183453_proto_calib_v2_sweep/COMMANDS.txt<br>IRRA-main/pack/20260429_183453_proto_calib_v2_sweep/run_proto_calib_v2_all.sh<br>IRRA-main/pack/20260429_183453_proto_calib_v2_sweep/summarize_proto_calib_v2.py<br>IRRA-main/processor/__init__.py<br>IRRA-main/processor/processor.py<br>IRRA-main/run_4_ablation_5ep_live_fixed.bat<br>IRRA-main/run_det_anchor_60epoch.bat<br>IRRA-main/run_gamma_schedule_20ep_realtime.bat<br>IRRA-main/run_hn_60ep_4exp.sh<br>IRRA-main/run_irra.sh<br>IRRA-main/run_mga_rde_screening_4exp_20ep.sh<br>IRRA-main/run_next_stage_experiments.bat<br>IRRA-main/run_relonly_vnext_20ep.bat<br>IRRA-main/run_relonly_vnext_20ep_realtime.bat<br>IRRA-main/run_rgbt_mum_ablations.sh<br>IRRA-main/run_stage_relonly_gamma005_20ep_realtime.bat<br>IRRA-main/run_strict_proto3_uspc_20ep.sh<br>IRRA-main/run_synthcap_yunwu_20ep_3exp.sh<br>IRRA-main/run_synthcap_yunwu_smoke.sh<br>IRRA-main/run_uspc_v1_ablation_60ep.sh<br>IRRA-main/run_uspc_v1_smoke_5ep.sh<br>IRRA-main/run_uspc_v2a_attr_smoke_2ep.sh<br>IRRA-main/run_uspc_v2b0_slot_local_20ep_grid.sh<br>IRRA-main/run_uspc_v2b0_slot_local_smoke_2ep.sh<br>IRRA-main/scripts/revision/README_AutoDL.md<br>IRRA-main/scripts/revision/run_new_degradation_matrix.sh<br>IRRA-main/scripts/revision/run_old_degradation_regression.sh<br>IRRA-main/solver/__init__.py<br>IRRA-main/solver/build.py<br>IRRA-main/solver/lr_scheduler.py<br>IRRA-main/sweep_lambda_unc_5ep.bat<br>IRRA-main/test.py<br>IRRA-main/tests/__init__.py<br>IRRA-main/tests/test_evaluation_degradation.py<br>IRRA-main/tests/test_revision_analysis_tools.py<br>IRRA-main/tools/analyze_qalr_v3_candidate_oracle.py<br>IRRA-main/tools/build_large_synth_bank_v2.py<br>IRRA-main/tools/build_large_synth_bank_v3.py<br>IRRA-main/tools/build_large_synth_bank_v4.py<br>IRRA-main/tools/build_large_synth_bank_v5_llm.py<br>IRRA-main/tools/build_mga_pseudo_from_captions.py<br>IRRA-main/tools/build_offline_hardneg_index.py<br>IRRA-main/tools/build_offline_prototype_bank.py<br>IRRA-main/tools/build_qalr_v4_gate_dataset.py<br>IRRA-main/tools/build_synth_bank_smoketest.py<br>IRRA-main/tools/build_synth_bank_v6_controlled.py<br>IRRA-main/tools/check_synth_bank_v6_controlled.py<br>IRRA-main/tools/check_synth_caption_bank.py<br>IRRA-main/tools/dump_rgbt_pedes_test_metadata.py<br>IRRA-main/tools/ensemble_eval_scores.py<br>IRRA-main/tools/revision/__init__.py<br>IRRA-main/tools/revision/color_group_evaluation.py<br>IRRA-main/tools/revision/color_terms_en.txt<br>IRRA-main/tools/revision/compare_score_files.py<br>IRRA-main/tools/revision/ensemble_scores.py<br>IRRA-main/tools/revision/feature_norm_analysis.py<br>IRRA-main/tools/revision/intensity_curve_summary.py<br>IRRA-main/tools/revision/noise_repeat_summary.py<br>IRRA-main/tools/revision/paired_seed_statistics.py<br>IRRA-main/tools/revision/score_io.py<br>IRRA-main/tools/revision/summarize_robustness.py<br>IRRA-main/tools/run_qalr_v2_safe_grid.sh<br>IRRA-main/tools/smoke_mga_dataloader.py<br>IRRA-main/tools/train_qalr_v4_gate.py<br>IRRA-main/tools/validate_mga_pseudo_json.py<br>IRRA-main/tools/weight_split_eval.py<br>IRRA-main/train.py<br>IRRA-main/utils/__init__.py<br>IRRA-main/utils/attribute_parser.py<br>IRRA-main/utils/checkpoint.py<br>IRRA-main/utils/comm.py<br>IRRA-main/utils/eval_degradation.py<br>IRRA-main/utils/iotools.py<br>IRRA-main/utils/logger.py<br>IRRA-main/utils/mdrc.py<br>IRRA-main/utils/meter.py<br>IRRA-main/utils/metrics.py<br>IRRA-main/utils/mtp_text.py<br>IRRA-main/utils/qalr_v4_gate.py<br>IRRA-main/utils/query_adaptive_rerank.py<br>IRRA-main/utils/simple_tokenizer.py<br>IRRA-main/utils/transport_rerank.py<br>IRRA-main/utils/uspc_text_views.py<br>IRRA-main/visualize.py |
| EOL + 允许追加 audit_out/ | 0 | OK | git ls-files -z（repo=E:\论文2修改\IRRA\IRRA-main-mraf）; 对每个候选执行 git show HEAD:<path>，并以 UTF-8 读取工作树后比较 CRLF/LF 归一化结果 |  |
| 其它语义差异 | 1 | WARN | git ls-files -z（repo=E:\论文2修改\IRRA\IRRA-main-mraf）; 对每个候选执行 git show HEAD:<path>，并以 UTF-8 读取工作树后比较 CRLF/LF 归一化结果 | .gitignore |
| 跳过/无法核验 | 0 | OK | git ls-files -z（repo=E:\论文2修改\IRRA\IRRA-main-mraf）; 对每个候选执行 git show HEAD:<path>，并以 UTF-8 读取工作树后比较 CRLF/LF 归一化结果 |  |

## 九、未解决项

| 检查项 | 值 | 状态 | 来源 | 备注 |
| --- | --- | --- | --- | --- |
| UNKNOWN 检查项 | 未找到 | OK | 本报告中 status 字段汇总 | 没有 UNKNOWN 项 |
