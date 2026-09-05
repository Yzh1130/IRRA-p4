## 一、探测到的标注文件与字段映射

| 标注/source | 顶层类型 | 记录数 | 身份字段 | 划分字段 | 文本字段 | 路径字段 | MD5 | status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| datasets/TextReID/RGBT-PEDES/RGBT-PEDES-Final/Caption.json | list | 4723 | id | split | captions | file_path | 97fde3821fc5a7a6ac896a3ac0dbf649 | OK |
| datasets/TextReID/RGBT-PEDES/RGBT-PEDES/caption.json | list | 4723 | id | split | captions | file_path | e053f2c58b4612e02519ead2fccdab0f | OK |
| datasets/TextReID/RSTPReid/data_captions.json | list | 20505 | id | split | captions | img_path | a40c29987c23fde17216fb5a9f84d8c9 | OK |

### 标注逐对逐字段比对

- `datasets/TextReID/RGBT-PEDES/RGBT-PEDES-Final/Caption.json` ↔ `datasets/TextReID/RGBT-PEDES/RGBT-PEDES/caption.json`：共享记录 4723，仅左 0，仅右 0，逐字段状态 WARN。 MD5(left)=97fde3821fc5a7a6ac896a3ac0dbf649，MD5(right)=e053f2c58b4612e02519ead2fccdab0f，字节相同=False。source=逐记录逐字段比较：datasets/TextReID/RGBT-PEDES/RGBT-PEDES-Final/Caption.json ↔ datasets/TextReID/RGBT-PEDES/RGBT-PEDES/caption.json；status=WARN

| 字段 | 值不同 | 仅左有 | 仅右有 | 不同记录键（前20） | source | status |
| --- | --- | --- | --- | --- | --- | --- |
| captions | 1459 | 0 | 0 | ['basename', '', '1488_1.jpg', 0, 0], ['basename', '', '1418_1.jpg', 0, 0], ['basename', '', '1668_1.jpg', 0, 0], ['basename', '', '1783_2.jpg', 0, 0], ['basename', '', '1386_1.jpg', 0, 0], ['basename', '', '1758_1.jpg', 0, 0], ['basename', '', '1377_3.jpg', 0, 0], ['basename', '', '1687_8.jpg', 0, 0], ['basename', '', '1639_4.jpg', 0, 0], ['basename', '', '1460_3.jpg', 0, 0], ['basename', '', '1729_5.jpg', 0, 0], ['basename', '', '1431_1.jpg', 0, 0], ['basename', '', '1742_2.jpg', 0, 0], ['basename', '', '1509_1.jpg', 0, 0], ['basename', '', '1410_3.jpg', 0, 0], ['basename', '', '1401_3.jpg', 0, 0], ['basename', '', '1777_2.jpg', 0, 0], ['basename', '', '1715_2.jpg', 0, 0], ['basename', '', '1639_3.jpg', 0, 0], ['basename', '', '1513_1.jpg', 0, 0] | 逐记录逐字段比较：datasets/TextReID/RGBT-PEDES/RGBT-PEDES-Final/Caption.json ↔ datasets/TextReID/RGBT-PEDES/RGBT-PEDES/caption.json | WARN |
| id | 4711 | 0 | 0 | ['basename', '', '1488_1.jpg', 0, 0], ['basename', '', '596_1.jpg', 0, 0], ['basename', '', '796_2.jpg', 0, 0], ['basename', '', '223_2.jpg', 0, 0], ['basename', '', '1418_1.jpg', 0, 0], ['basename', '', '1668_1.jpg', 0, 0], ['basename', '', '1023_2.jpg', 0, 0], ['basename', '', '1783_2.jpg', 0, 0], ['basename', '', '787_1.jpg', 0, 0], ['basename', '', '1386_1.jpg', 0, 0], ['basename', '', '1758_1.jpg', 0, 0], ['basename', '', '1377_3.jpg', 0, 0], ['basename', '', '1687_8.jpg', 0, 0], ['basename', '', '139_2.jpg', 0, 0], ['basename', '', '807_2.jpg', 0, 0], ['basename', '', '816_4.jpg', 0, 0], ['basename', '', '355_2.jpg', 0, 0], ['basename', '', '413_2.jpg', 0, 0], ['basename', '', '1367_1.jpg', 0, 0], ['basename', '', '431_1.jpg', 0, 0] | 逐记录逐字段比较：datasets/TextReID/RGBT-PEDES/RGBT-PEDES-Final/Caption.json ↔ datasets/TextReID/RGBT-PEDES/RGBT-PEDES/caption.json | WARN |
| processed_tokens | 1 | 0 | 0 | ['basename', '', '1527_1.jpg', 0, 0] | 逐记录逐字段比较：datasets/TextReID/RGBT-PEDES/RGBT-PEDES-Final/Caption.json ↔ datasets/TextReID/RGBT-PEDES/RGBT-PEDES/caption.json | WARN |

### 仓库 loader 的解析优先级推断

| 项目 | 值 | source | status |
| --- | --- | --- | --- |
| 传给 loader 的 root | E:\论文2修改\IRRA\IRRA-main-mraf\IRRA-main\datasets\TextReID | datasets/rgbtpedes.py | OK |
| data_dir 候选顺序 | E:\论文2修改\IRRA\IRRA-main-mraf\IRRA-main\datasets\TextReID\RGBT-PEDES\RGBT-PEDES-Final → E:\论文2修改\IRRA\IRRA-main-mraf\IRRA-main\datasets\TextReID\RGBT-PEDES\RGBT-PEDES → E:\论文2修改\IRRA\IRRA-main-mraf\IRRA-main\datasets\TextReID\RGBT-PEDES | datasets/rgbtpedes.py | OK |
| 推断选中 data_dir | E:\论文2修改\IRRA\IRRA-main-mraf\IRRA-main\datasets\TextReID\RGBT-PEDES\RGBT-PEDES-Final | datasets/rgbtpedes.py | OK |
| 标注文件名顺序 | Caption.json → caption.json | datasets/rgbtpedes.py | OK |
| 推断实际加载标注 | E:\论文2修改\IRRA\IRRA-main-mraf\IRRA-main\datasets\TextReID\RGBT-PEDES\RGBT-PEDES-Final\Caption.json | datasets/rgbtpedes.py | OK |

## 二、身份、图像对与文本统计表

| 划分 | 原始唯一身份 | 重编号标签数 | 标注记录 | 具备两侧派生路径 | 磁盘实际配对 | 文本描述 | 每记录 caption 分布 | source | status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| train | 1266 | 1266 | 3250 | 3250 | 3250 | 5497 | {1: 1003, 2: 2247} | datasets/TextReID/RGBT-PEDES/RGBT-PEDES-Final/Caption.json | OK |
| test | 556 | 556 | 1473 | 1473 | 1473 | 2490 | {1: 456, 2: 1017} | datasets/TextReID/RGBT-PEDES/RGBT-PEDES-Final/Caption.json | OK |
| 合计 | 1822 | 各 split 独立编号 | 4723 | 4723 | 4723 | 7987 | {2: 3264, 1: 1459} | datasets/TextReID/RGBT-PEDES/RGBT-PEDES-Final/Caption.json | OK |

| 项目 | 值 | source | status |
| --- | --- | --- | --- |
| train ∩ test 身份 | 无 | datasets/TextReID/RGBT-PEDES/RGBT-PEDES-Final/Caption.json | OK |
| train ∪ test 身份数 | 1822 | datasets/TextReID/RGBT-PEDES/RGBT-PEDES-Final/Caption.json | OK |
| 原始身份 ID 最小值 | 1 | datasets/TextReID/RGBT-PEDES/RGBT-PEDES-Final/Caption.json | OK |
| 原始身份 ID 最大值 | 1822 | datasets/TextReID/RGBT-PEDES/RGBT-PEDES-Final/Caption.json | OK |
| caption/记录 最小值 | 1 | datasets/TextReID/RGBT-PEDES/RGBT-PEDES-Final/Caption.json | OK |
| caption/记录 中位数 | 2 | datasets/TextReID/RGBT-PEDES/RGBT-PEDES-Final/Caption.json | OK |
| caption/记录 最大值 | 2 | datasets/TextReID/RGBT-PEDES/RGBT-PEDES-Final/Caption.json | OK |

## 三、本次实测 vs 已确证基线

| 项目 | 本次实测 | 已确证基线 | 是否一致 | source | status |
| --- | --- | --- | --- | --- | --- |
| 总记录 | 4723 | 4723 | 逐值一致 | datasets/TextReID/RGBT-PEDES/RGBT-PEDES-Final/Caption.json | OK |
| ID 唯一数 | 1822 | 1822 | 逐值一致 | datasets/TextReID/RGBT-PEDES/RGBT-PEDES-Final/Caption.json | OK |
| train 身份 | 1266 | 1266 | 逐值一致 | datasets/TextReID/RGBT-PEDES/RGBT-PEDES-Final/Caption.json | OK |
| train 记录 | 3250 | 3250 | 逐值一致 | datasets/TextReID/RGBT-PEDES/RGBT-PEDES-Final/Caption.json | OK |
| train caption | 5497 | 5497 | 逐值一致 | datasets/TextReID/RGBT-PEDES/RGBT-PEDES-Final/Caption.json | OK |
| test 身份 | 556 | 556 | 逐值一致 | datasets/TextReID/RGBT-PEDES/RGBT-PEDES-Final/Caption.json | OK |
| test 记录 | 1473 | 1473 | 逐值一致 | datasets/TextReID/RGBT-PEDES/RGBT-PEDES-Final/Caption.json | OK |
| test caption | 2490 | 2490 | 逐值一致 | datasets/TextReID/RGBT-PEDES/RGBT-PEDES-Final/Caption.json | OK |
| train∩test 身份数 | 0 | 0 | 逐值一致 | datasets/TextReID/RGBT-PEDES/RGBT-PEDES-Final/Caption.json | OK |
| caption 数分布 | {2: 3264, 1: 1459} | {1: 1459, 2: 3264} | 逐值一致 | datasets/TextReID/RGBT-PEDES/RGBT-PEDES-Final/Caption.json | OK |

基线名称=RGBT-PEDES；source=任务给定已确证基线。

## 四、异常记录清单

| 项目 | 值 | source | status |
| --- | --- | --- | --- |
| RGB 目录文件数 | 4723 | E:\论文2修改\IRRA\IRRA-main-mraf\IRRA-main\datasets\TextReID\RGBT-PEDES\RGBT-PEDES-Final\RGB | OK |
| T 目录文件数 | 4723 | E:\论文2修改\IRRA\IRRA-main-mraf\IRRA-main\datasets\TextReID\RGBT-PEDES\RGBT-PEDES-Final\T | OK |
| RGB/T 同名配对文件数 | 4723 | E:\论文2修改\IRRA\IRRA-main-mraf\IRRA-main\datasets\TextReID\RGBT-PEDES\RGBT-PEDES-Final\RGB；E:\论文2修改\IRRA\IRRA-main-mraf\IRRA-main\datasets\TextReID\RGBT-PEDES\RGBT-PEDES-Final\T | OK |

| 检查项 | 记录/文件数 | 身份 / basename / 预期路径 | source | status |
| --- | --- | --- | --- | --- |
| 缺身份字段 | 0 | 无 | datasets/TextReID/RGBT-PEDES/RGBT-PEDES-Final/Caption.json | OK |
| 缺 split | 0 | 无 | datasets/TextReID/RGBT-PEDES/RGBT-PEDES-Final/Caption.json | OK |
| 无路径标注 | 0 | 无 | datasets/TextReID/RGBT-PEDES/RGBT-PEDES-Final/Caption.json | OK |
| 仅 RGB 路径标注 | 0 | 无 | datasets/TextReID/RGBT-PEDES/RGBT-PEDES-Final/Caption.json | OK |
| 仅 T 路径标注 | 0 | 无 | datasets/TextReID/RGBT-PEDES/RGBT-PEDES-Final/Caption.json | OK |
| RGB 文件未找到 | 0 | 无 | datasets/TextReID/RGBT-PEDES/RGBT-PEDES-Final/Caption.json | OK |
| T 文件未找到 | 0 | 无 | datasets/TextReID/RGBT-PEDES/RGBT-PEDES-Final/Caption.json | OK |
| 无描述 | 1 | 1527 | datasets/TextReID/RGBT-PEDES/RGBT-PEDES-Final/Caption.json | WARN |
| 磁盘仅 RGB basename | 0 | 无 | E:\论文2修改\IRRA\IRRA-main-mraf\IRRA-main\datasets\TextReID\RGBT-PEDES\RGBT-PEDES-Final\RGB；E:\论文2修改\IRRA\IRRA-main-mraf\IRRA-main\datasets\TextReID\RGBT-PEDES\RGBT-PEDES-Final\T | OK |
| 磁盘仅 T basename | 0 | 无 | E:\论文2修改\IRRA\IRRA-main-mraf\IRRA-main\datasets\TextReID\RGBT-PEDES\RGBT-PEDES-Final\RGB；E:\论文2修改\IRRA\IRRA-main-mraf\IRRA-main\datasets\TextReID\RGBT-PEDES\RGBT-PEDES-Final\T | OK |
| 成对文件未见于标注 | 0 | 无 | E:\论文2修改\IRRA\IRRA-main-mraf\IRRA-main\datasets\TextReID\RGBT-PEDES\RGBT-PEDES-Final\RGB；E:\论文2修改\IRRA\IRRA-main-mraf\IRRA-main\datasets\TextReID\RGBT-PEDES\RGBT-PEDES-Final\T | OK |
| 任一 RGB/T/描述计数为 0 的身份 | 0 | 无 | datasets/TextReID/RGBT-PEDES/RGBT-PEDES-Final/Caption.json | OK |

## 五、loader 内省结果

### 四项专项事实（精确行号与原文）

| 事实 | 源码原文 | source | status |
| --- | --- | --- | --- |
| 数据目录解析优先级 | 49:         candidates = [<br>50:             op.join(dataset_dir, "RGBT-PEDES-Final"),<br>51:             op.join(dataset_dir, "RGBT-PEDES"),<br>52:             dataset_dir,<br>53:         ] | datasets/rgbtpedes.py:49 | OK |
| 标注文件名解析优先级 | 60:         for fname in ("Caption.json", "caption.json"): | datasets/rgbtpedes.py:60 | OK |
| val 为空时复用 test | 80:         if len(val_annos) == 0:<br>81:             val_annos = test_annos | datasets/rgbtpedes.py:80 | OK |
| 每个 split 独立身份重编号 | 92:         raw_pids = sorted({int(anno["id"]) for anno in annos})<br>93:         pid2label = {pid: idx for idx, pid in enumerate(raw_pids)} | datasets/rgbtpedes.py:92 | OK |
| rgbtpedes.py 全文过滤/跳过逻辑 | continue/filter/skip 匹配数=0；len(...)==0 命中：80: if len(val_annos) == 0:；81: val_annos = test_annos；其下一条语句 continue/return 命中数=0 | datasets/rgbtpedes.py:1-148；正则 continue\|filter\|skip 及 if.*len(...)==0 下一条语句 continue/return | OK |
| ID 分类头 num_classes | len(dataset.train_id_container)；推断值=1266 | datasets/build.py:451 | OK |
| 训练样本按 caption 展开 | 原文=                for caption in anno["captions"]:；推断展开样本数=5497 | datasets/rgbtpedes.py:104 | OK |

### 原始 ID 到重编号标签

| split | 连续整数 | 映射前20条 | 映射总数 | source | status |
| --- | --- | --- | --- | --- | --- |
| test | True；范围=0..555 | [('7', 0), ('8', 1), ('9', 2), ('10', 3), ('12', 4), ('15', 5), ('17', 6), ('22', 7), ('25', 8), ('26', 9), ('29', 10), ('33', 11), ('39', 12), ('41', 13), ('42', 14), ('43', 15), ('44', 16), ('54', 17), ('61', 18), ('62', 19)] | 556 | datasets/TextReID/RGBT-PEDES/RGBT-PEDES-Final/Caption.json | OK |
| train | True；范围=0..1265 | [('1', 0), ('2', 1), ('3', 2), ('4', 3), ('5', 4), ('6', 5), ('11', 6), ('13', 7), ('14', 8), ('16', 9), ('18', 10), ('19', 11), ('20', 12), ('21', 13), ('23', 14), ('24', 15), ('27', 16), ('28', 17), ('30', 18), ('31', 19)] | 1266 | datasets/TextReID/RGBT-PEDES/RGBT-PEDES-Final/Caption.json | OK |

### Dataset 对象 best-effort 内省

| 检查项 | 值 | source | status |
| --- | --- | --- | --- |
| 预先搜索的数据构建候选模块 | datasets.bases, datasets.build, datasets.cuhkpedes, datasets.icfgpedes, datasets.rgbtpedes, datasets.rstpreid, utils.metrics, utils.mtp_text, utils.options, utils.qalr_v4_gate, utils.query_adaptive_rerank, utils.transport_rerank, utils.uspc_text_views | datasets/、data/、utils/ 静态扫描；逐项 source 见 audit_out/dataset_probe.json | OK |
| 构造 Dataset 状态 | OK | import datasets.rgbtpedes.RGBTPEDES(root=E:\论文2修改\IRRA\IRRA-main-mraf\IRRA-main\datasets\TextReID, verbose=False) | OK |
| train dataset 长度 | 5497 | import datasets.rgbtpedes.RGBTPEDES(root=E:\论文2修改\IRRA\IRRA-main-mraf\IRRA-main\datasets\TextReID, verbose=False) | OK |
| test gallery/image_pids 长度 | 1473 | import datasets.rgbtpedes.RGBTPEDES(root=E:\论文2修改\IRRA\IRRA-main-mraf\IRRA-main\datasets\TextReID, verbose=False) | OK |
| test query/caption_pids 长度 | 2490 | import datasets.rgbtpedes.RGBTPEDES(root=E:\论文2修改\IRRA\IRRA-main-mraf\IRRA-main\datasets\TextReID, verbose=False) | OK |
| train 实际唯一 label 数 | 1266 | import datasets.rgbtpedes.RGBTPEDES(root=E:\论文2修改\IRRA\IRRA-main-mraf\IRRA-main\datasets\TextReID, verbose=False) | OK |
| test 实际唯一 label 数 | 556 | import datasets.rgbtpedes.RGBTPEDES(root=E:\论文2修改\IRRA\IRRA-main-mraf\IRRA-main\datasets\TextReID, verbose=False) | OK |
| train label 连续 | True | import datasets.rgbtpedes.RGBTPEDES(root=E:\论文2修改\IRRA\IRRA-main-mraf\IRRA-main\datasets\TextReID, verbose=False) | OK |
| test label 连续 | True | import datasets.rgbtpedes.RGBTPEDES(root=E:\论文2修改\IRRA\IRRA-main-mraf\IRRA-main\datasets\TextReID, verbose=False) | OK |
| 内省说明 | 仅构造 RGBTPEDES；未构造 DataLoader、未索引 dataset、未执行 forward。 | import datasets.rgbtpedes.RGBTPEDES(root=E:\论文2修改\IRRA\IRRA-main-mraf\IRRA-main\datasets\TextReID, verbose=False) | OK |

### 数据构建路径中可能的过滤/跳过行（仅列事实）

| source | 原文 | status |
| --- | --- | --- |
| datasets/bases.py:125 |                 continue | OK |
| datasets/bases.py:140 |                 continue | OK |
| datasets/bases.py:143 |                 continue | OK |
| datasets/bases.py:146 |                 continue | OK |
| datasets/bases.py:344 |                 continue | OK |
| datasets/bases.py:347 |                 continue | OK |
| datasets/bases.py:350 |                 continue | OK |
| datasets/bases.py:353 |                 continue | OK |
| datasets/build.py:187 |             continue | OK |
| datasets/build.py:191 |                 continue | OK |
| datasets/build.py:194 |                 continue | OK |
| datasets/build.py:197 |                 continue | OK |
| datasets/build.py:305 |             continue | OK |
| datasets/build.py:316 |             continue | OK |
| datasets/build.py:391 |                 continue | OK |
| datasets/build.py:400 |                     continue | OK |
| datasets/build.py:403 |                     continue | OK |
| datasets/mga_pseudo.py:135 |             continue | OK |
| datasets/mga_pseudo.py:280 |                 continue | OK |
| datasets/mga_pseudo.py:293 |                 continue | OK |
| datasets/mga_pseudo.py:316 |             continue | OK |
| datasets/mga_pseudo.py:319 |             continue | OK |
| datasets/mga_pseudo.py:322 |             continue | OK |
| datasets/preprocessing.py:28 |         # Skip augmentation by probability. | OK |
| datasets/rgbtpedes.py:80 |         if len(val_annos) == 0: | OK |
| datasets/sampler.py:65 |                 if len(batch_idxs_dict[pid]) == 0: | OK |

### `model/mraf.py` 参数静态清点

| 作用域 | 参数 | 默认值 | source | status |
| --- | --- | --- | --- | --- |
| ModalityReliabilityAdaptiveFusion.__init__ | feature_dim | 无默认值 | model/mraf.py:64 | OK |
| ModalityReliabilityAdaptiveFusion.__init__ | hidden_dim | 128 | model/mraf.py:65 | OK |
| ModalityReliabilityAdaptiveFusion.__init__ | dropout | 0.1 | model/mraf.py:66 | OK |
| ModalityReliabilityAdaptiveFusion.__init__ | gate_temperature | 1.0 | model/mraf.py:67 | OK |
| ModalityReliabilityAdaptiveFusion.__init__ | gate_logit_scale | 1.0 | model/mraf.py:68 | OK |
| ModalityReliabilityAdaptiveFusion.__init__ | gate_fp32 | False | model/mraf.py:69 | OK |
| ModalityReliabilityAdaptiveFusion.__init__ | gate_base_grad_scale | 1.0 | model/mraf.py:70 | OK |
| ModalityReliabilityAdaptiveFusion.__init__ | equalize_modality_norm | False | model/mraf.py:71 | OK |
| ModalityReliabilityAdaptiveFusion.__init__ | freeze_gate | False | model/mraf.py:72 | OK |
| compute_mraf_reliability_targets | clean_rgb | 无默认值 | model/mraf.py:276 | OK |
| compute_mraf_reliability_targets | clean_ir | 无默认值 | model/mraf.py:277 | OK |
| compute_mraf_reliability_targets | degraded_rgb | 无默认值 | model/mraf.py:278 | OK |
| compute_mraf_reliability_targets | degraded_ir | 无默认值 | model/mraf.py:279 | OK |
| compute_mraf_reliability_targets | target_temperature | 0.1 | model/mraf.py:280 | OK |
| compute_mraf_reliability_targets | eps | 1e-08 | model/mraf.py:281 | OK |
| resolve_mraf_rel_lambda | base_lambda | 无默认值 | model/mraf.py:339 | OK |
| resolve_mraf_rel_lambda | warmup_epoch | 无默认值 | model/mraf.py:340 | OK |
| resolve_mraf_rel_lambda | zero_based_epoch | 无默认值 | model/mraf.py:341 | OK |
| utils/options.py CLI（传入 model/mraf.py 或训练逻辑） | mraf_enabled | False | utils/options.py:123 | OK |
| utils/options.py CLI（传入 model/mraf.py 或训练逻辑） | mraf_hidden_dim | 128 | utils/options.py:124 | OK |
| utils/options.py CLI（传入 model/mraf.py 或训练逻辑） | mraf_dropout | 0.1 | utils/options.py:125 | OK |
| utils/options.py CLI（传入 model/mraf.py 或训练逻辑） | mraf_gate_temperature | 1.0 | utils/options.py:126 | OK |
| utils/options.py CLI（传入 model/mraf.py 或训练逻辑） | mraf_gate_logit_scale | 1.0 | utils/options.py:127 | OK |
| utils/options.py CLI（传入 model/mraf.py 或训练逻辑） | mraf_gate_fp32 | False | utils/options.py:128 | OK |
| utils/options.py CLI（传入 model/mraf.py 或训练逻辑） | mraf_gate_base_grad_scale | 1.0 | utils/options.py:129 | OK |
| utils/options.py CLI（传入 model/mraf.py 或训练逻辑） | mraf_equalize_modality_norm | False | utils/options.py:130 | OK |
| utils/options.py CLI（传入 model/mraf.py 或训练逻辑） | mraf_freeze_gate | False | utils/options.py:131 | OK |
| utils/options.py CLI（传入 model/mraf.py 或训练逻辑） | mraf_scorer_lr_multiplier | 1.0 | utils/options.py:132 | OK |
| utils/options.py CLI（传入 model/mraf.py 或训练逻辑） | mraf_rel_lambda | 0.01 | utils/options.py:133 | OK |
| utils/options.py CLI（传入 model/mraf.py 或训练逻辑） | mraf_target_temperature | 0.1 | utils/options.py:134 | OK |
| utils/options.py CLI（传入 model/mraf.py 或训练逻辑） | mraf_rel_warmup_epoch | 3 | utils/options.py:135 | OK |
| utils/options.py CLI（传入 model/mraf.py 或训练逻辑） | mraf_balance_lambda | 0.0 | utils/options.py:136 | OK |

## 六、query/gallery 口径

| 检查项 | 值 | source | status |
| --- | --- | --- | --- |
| gallery/image_pids 数量 | 1473 | datasets/TextReID/RGBT-PEDES/RGBT-PEDES-Final/Caption.json | OK |
| query/caption_pids 数量 | 2490 | datasets/TextReID/RGBT-PEDES/RGBT-PEDES-Final/Caption.json | OK |
| query（文本）唯一身份 | 556 | datasets/TextReID/RGBT-PEDES/RGBT-PEDES-Final/Caption.json | OK |
| gallery（图像）唯一身份 | 556 | datasets/TextReID/RGBT-PEDES/RGBT-PEDES-Final/Caption.json | OK |
| 两侧是否相等 | True | datasets/TextReID/RGBT-PEDES/RGBT-PEDES-Final/Caption.json | OK |
| query - gallery | 无 | datasets/TextReID/RGBT-PEDES/RGBT-PEDES-Final/Caption.json | OK |
| gallery - query | 无 | datasets/TextReID/RGBT-PEDES/RGBT-PEDES-Final/Caption.json | OK |

## 七、未解决项

| 检查项 | 值 | source | 说明 |
| --- | --- | --- | --- |
| 指定标注文件：caption_2_no_color.json | 未找到 | E:\论文2修改\IRRA\IRRA-main-mraf\IRRA-main\datasets\TextReID | 用于三份标注的逐对逐字段复核；缺失时保留 UNKNOWN，供服务器数据目录复现 |

## 附录：原始探测输出路径

| 文件 | source | status |
| --- | --- | --- |
| E:\论文2修改\IRRA\IRRA-main-mraf\IRRA-main\audit_out\stage0_selftest\rgbt\dataset_probe.json | E:\论文2修改\IRRA\IRRA-main-mraf\IRRA-main\audit_out\stage0_selftest\rgbt\dataset_probe.json | OK |
