#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""论文4 第十八轮（RGBNT201-PEDES 外部验证）：L2 结果的机械判读。

依据《论文4_第十八轮_RGBNT201_数据结论与冻结规格》v1：§3.3 O3/O4（去重子集与措辞）、§3.6 统计、§3.7 读法与分档、
§3.8 有效性检查、§3.9 跨相机子分析、§8 第 1 条；E4 的结局表与三条限制沿用《论文4_第十七轮_独立裁决与收尾计划》§3.6。
L2 开跑前冻结，结果出来后不改；脚本与规格文字冲突时以规格为准，并报告冲突。

输入：p4_r18.sh 写出的 DIGEST18_L2.txt，须在 AutoDL 上直接读该文件（经终端复制、改过换行或加了 BOM 的文本会被拒绝）。
只读其中三类行：
  - STATUS_L1= / STATUS_L2= 行；
  - 末尾的 DIGEST_STATS 行（核对摘要完整、没有被截断或拼接）；
  - 以 "R18J " 开头的行，即 R18J_core（r18_core.py 的输出）。
不读特征、不重算任何评分；判定只用各种子的未舍入值；只用 Python 标准库。

判读（每条都须在 3 个种子上都成立）：
  E1 A 组 QE+M − QE-DBA 的 mAP 差 ≥ +1.0，且 95% 区间下界 > 0；
  E2 B 组 QE+M − QE-DBA 的 mAP 差 ≥ −0.3，且 R1 差 ≥ −0.3（点估计）；
  E3 A 组 QE+M − 该种子测试集上 A 组最强的通用后处理 ≥ +1.0 且下界 > 0；B 组 QE+M − B 组最强者 ≥ −0.3；
     候选池 KR、RVN、QE-DBA、KRir、RVNir、QEir、GP-same；
  E4 QE+M − QE+Const 按第十七轮 §3.6 的结局表（①a / ①b / ② / ③ / 未覆盖）机械判读。
分档（由保守到宽松）：实现或协议问题（任一有效性检查不通过，或 SELECTION_EQUAL=false）、
  未见效果（三个种子 E1 的点估计都 ≤ 0）、有限适用（其余）、有效（E1、E2 都成立）。
E3、E4 与跨相机子分析只限定措辞，不改变档次。做了去重子集时，论文取全量与子集两档中较保守的一档；
本规格的子集有 26 个身份（≥ 10），照常判档。两个范围的 E2 写法、E3、E4 不一致时，措辞同样取较保守者
（规格 §3.7，L2 前补写）：E2 取较差的写法；E3 两个范围都成立才写成立；E4 由保守到宽松为 ③、②、①b、①a，
任一范围为"未覆盖"时另行裁决；跨相机子分析只在全量上做。

用法：python3 r18_verdict.py DIGEST18_L2.txt
退出码：0 已判档；2 不判定（输入不完整、不一致，或去重子集缺失）；1 用法或读取错误；3 脚本内部错误。
"""
import hashlib, json, math, os, re, sys, traceback, unicodedata
from decimal import Decimal, ROUND_HALF_UP

SEEDS = (101, 102, 103)
FITS = (7001, 7002, 7003)
SCOPES = ('full', 'subset')
SCOPE_ZH = {'full': '全量测试集（30 个身份）', 'subset': '去重测试子集（26 个身份）'}
N_ID = {'full': 30, 'subset': 26}        # 规格 §3.2：测试 30 个身份；§3.3：测试侧重叠 4 个，去重后 26 个
N_Q_FULL = 836                           # 规格 §3.2：测试 836 条描述，没有空白描述
E1_GAIN, TOL = 1.0, -0.3                 # E1/E3 的 +1.0；E2/E3 与第十七轮的原容差 −0.3
C4_A, C4_B, C5_B, C5_A = 0.5, -0.1, 0.3, -0.3
THRESHOLDS = (E1_GAIN, 0.0, TOL, C4_A, C4_B, C5_B)
POOL = ('KR', 'RVN', 'QE-DBA', 'KRir', 'RVNir', 'QEir', 'GP-same')
RUNNER_HEADS = {'anyh', 'srci', 'gateC'}  # 运行器只允许这三个辅助头的单类拟合改为常数（规格 §4 第 9 条）
RUNNER_PARTS = {'full': ('constrained', 'const', 'xcam'), 'subset': ('constrained', 'const')}
LOC_METHODS = ('QE+M', 'QE+Const', 'QE+NoGate', 'QE+CAET', 'QE+EM', 'QE+SRC2f')   # r17_const 的非前端方法
CONST_GRID = (0.0, 0.05, 0.1, 0.2, 0.25, 0.5, 0.75, 1.0)                           # 第十七轮 γ 网格
REPRO_TOL = 1e-6                         # r17 对 r15 共有方法的复现容差（规格 §3.8）
TIER_ORDER = ('实现或协议问题', '未见效果', '有限适用', '有效')   # 由保守到宽松
E4_ORDER = ('③', '②', '①b', '①a')                              # 由保守到宽松
E2_ORDER = ('超出原容差', '点估计在原容差内，但不能排除超出容差的损害', '损害在原容差以内')   # 由保守到宽松
MIXES = ('M0', 'M4', 'M6c', 'M1g', 'M4c60', 'M9', 'M5e', 'M5f', 'M4id', 'M10', 'M4c25', 'M4c50', 'M4c75')
EQ_TOL = 1e-9
NAN = float('nan')

# 第十七轮 §3.6 结局表"论文写法"一列（照录）；本轮只限定第二数据集上关于门控的措辞，
# RGBT-PEDES 上第十七轮的结局 ①a 与 C1–C5 不变（规格 §3.1）
E4_WORDING = {
    '①a': '写"相对开发集选定的全局常数增益，门控在 X 组带来 +a [lo, hi] 的增量（三个种子），达到预注册门控/迁移比较判据的形式"；'
          '另一组按第一条限制写',
    '①b': '写"可重复但小于预注册判据幅度的增量 +a [lo, hi]"；不写"必要"；另一组按第一条限制写',
    '②': '写"门控以 X 组的 +a 换来 Y 组的 −b，后者超出原容差"，两侧都给区间；不写门控必要、更优或更安全',
    '③': '不能写门控有独立价值（第十七轮表中"方法退为……"一句针对 RGBT-PEDES 的定稿；本轮只限定第二数据集上的措辞）',
    '未覆盖': '两组都可重复为正，以 A、B 分别作得益组时一条读成①、另一条读成②，§3.6 的两行同时字面成立；'
             '脚本不替你选，另行裁决，裁决前不按任一版写',
}
SECOND_LIMIT = ('有权衡不等于创新足够：门控能否作为方法贡献，要看得到的一侧是否在三个种子上可重复、幅度与原判据形式相比如何、'
                '失去的一侧是否在原容差以内；不能因为"存在权衡"就写成门控有独立价值')
TIER_SENTENCE = {   # 规格 §8 第 1 条（"有效"一句按第十七轮第一条限制补写 B 组的写法，L2 前修订）
    '有效': '写"在 RGBNT201-PEDES 上复现了 A 组增益"，B 组的代价按 E2 的写法写',
    '有限适用': '写清哪一条不成立（E1 或 E2，逐种子给出数字）',
    '未见效果': '如实写未复现，并在讨论中分析原因',
    '实现或协议问题': '先排查实现与数据，查明之前不作方法结论；不改方法、不换划分、不追加测试集调参',
}


# ============================================================== 小工具
def num(x):
    """有限实数才返回 float；布尔、None、NaN、字符串一律返回 None"""
    if isinstance(x, bool) or not isinstance(x, (int, float)):
        return None
    x = float(x)
    return x if math.isfinite(x) else None


def is_int(x):
    return isinstance(x, int) and not isinstance(x, bool)


def is_nan(x):
    return isinstance(x, float) and math.isnan(x)


def h2(x):
    """两位小数，ROUND_HALF_UP，与 r18_core 的 mean_2dp 同一做法；非有限或量级过大时原样给出"""
    try:
        x = float(x)
        if not math.isfinite(x) or abs(x) >= 1e15:
            return repr(x)
        return str(Decimal(str(x)).quantize(Decimal('0.01'), rounding=ROUND_HALF_UP))
    except (TypeError, ValueError, ArithmeticError):
        return '?'


def sg(x):
    """带符号的两位小数（表格用）"""
    if x is None or is_nan(x):
        return '—'
    s = h2(x)
    return '+' + s if s[:1].isdigit() else s


def sgx(x):
    """判读用：带符号的两位小数；离任一判读门槛不到 0.005（两位小数可能与门槛相同）时，括号内附未舍入值的完整表示"""
    s = sg(x)
    try:
        if s != '—' and any(abs(float(x) - t) < 0.005 for t in THRESHOLDS):
            s += f'({float(x)!r})'
    except (TypeError, ValueError, OverflowError):
        pass
    return s


def ci(p, lo, hi):
    return f'{sg(p)} [{sg(lo)}, {sg(hi)}]'


def cix(p, lo, hi):
    """判读用：点估计与下界参与判定，附未舍入值；上界只显示"""
    return f'{sgx(p)} [{sgx(lo)}, {sg(hi)}]'


def yes(b):
    return '✓' if b is True else ('✗' if b is False else '?')


def dwidth(s):
    return sum(2 if unicodedata.east_asian_width(ch) in 'WF' else 1 for ch in str(s))


def pad(s, width, right=False):
    s = str(s)
    fill = ' ' * max(0, width - dwidth(s))
    return fill + s if right else s + fill


def center(s, width):
    s = str(s)
    gap = max(0, width - dwidth(s))
    return ' ' * (gap // 2) + s + ' ' * (gap - gap // 2)


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def as_dict(v):
    return v if isinstance(v, dict) else {}


# ============================================================== 解析
RX_STATS = re.compile(r'DIGEST_STATS lines=(\d+) bytes=(\d+)')
RX_STATUS = re.compile(r'STATUS_(L1|L2)=(.*)')


def parse(text):
    """返回 dict(problems, status, rows, n_r18j)。problems 非空即不判定。"""
    problems = []
    lines = text.split('\n')
    stats = [ln for ln in lines if ln.startswith('DIGEST_STATS')]
    if len(stats) != 1:
        problems.append(f'DIGEST_STATS 行应恰有 1 行，实有 {len(stats)} 行（摘要缺尾或多份拼接）')
    else:
        m = RX_STATS.fullmatch(stats[0])
        if not m:
            problems.append('DIGEST_STATS 行格式不对：' + stats[0][:80])
        else:
            n_lines, n_bytes = int(m.group(1)), int(m.group(2))
            if not text.endswith(stats[0] + '\n'):
                problems.append('DIGEST_STATS 不是摘要的最后一行')
            if text.count('\n') != n_lines or len(text.encode('utf-8')) != n_bytes:
                problems.append(f'DIGEST_STATS 记的是 {n_lines} 行 / {n_bytes} 字节，实际为 {text.count(chr(10))} 行 / '
                                f'{len(text.encode("utf-8"))} 字节（摘要被截断或改动；须直接读 AutoDL 上的文件）')
    status = {'L1': [], 'L2': []}
    for ln in lines:
        m = RX_STATUS.fullmatch(ln)
        if m:
            status[m.group(1)].append(m.group(2).strip())
    for ph in ('L1', 'L2'):
        if not status[ph]:
            problems.append(f'没有 STATUS_{ph} 行')
        elif any(v != 'COMPLETE' for v in status[ph]):
            problems.append(f'STATUS_{ph} 不是 COMPLETE：' + ' / '.join(status[ph]))
    rows = {}
    n_r18j = 0
    for i, ln in enumerate(lines, 1):
        if not ln.startswith('R18J '):
            continue
        n_r18j += 1
        try:
            row = json.loads(ln[5:])
        except ValueError:
            problems.append(f'第 {i} 行 R18J 不是合法 JSON')
            continue
        if not isinstance(row, dict) or set(row) != {'src', 'set', 'seed', 'key', 'v'}:
            problems.append(f'第 {i} 行 R18J 字段不对')
            continue
        seed = row['seed']
        if not (seed == 'mean' or (is_int(seed) and seed in SEEDS)):
            problems.append(f'第 {i} 行 R18J 的种子不在 {SEEDS} 内：{seed!r}')
            continue
        if row['set'] not in SCOPES:
            problems.append(f'第 {i} 行 R18J 的范围不认识：{row["set"]!r}')
            continue
        if not isinstance(row['src'], str) or not isinstance(row['key'], str):
            problems.append(f'第 {i} 行 R18J 的 src/key 不是字符串')
            continue
        k = (row['src'], row['set'], seed, row['key'])
        if k in rows:
            problems.append(f'R18J 行重复：{k}（摘要可能含多次运行）')
            continue
        rows[k] = row['v']
    if n_r18j == 0:
        problems.append('没有任何 R18J 行（L2 没有跑完，或读的不是 DIGEST18_L2）')
    return dict(problems=problems, status=status, rows=rows, n_r18j=n_r18j)


# ============================================================== 单个范围（全量 / 子集）的判读
def tol_wording(points, lows):
    """第十七轮 §3.6 第一条限制：点估计与区间下界对照 −0.3"""
    if any(p < TOL for p in points):
        return E2_ORDER[0]
    if any(l < TOL for l in lows):
        return E2_ORDER[1]
    return E2_ORDER[2]


def e4_outcome(trip):
    """trip(seed, group, metric) -> (估计, 下界, 上界)，即 QE+M − QE+Const。返回 (结局, 路径, 过程)。"""
    def is_pos(t):
        return t[0] > 0 and t[1] > 0          # 点估计为正且区间不含 0
    P = {g: all(is_pos(trip(s, g, 'AP')) for s in SEEDS) for g in 'AB'}
    T = {g: all(trip(s, g, 'AP')[0] >= TOL and trip(s, g, 'R1')[0] >= TOL for s in SEEDS) for g in 'AB'}
    F4 = (all(is_pos(trip(s, 'A', 'AP')) and trip(s, 'A', 'AP')[0] >= C4_A for s in SEEDS)
          and all(trip(s, 'B', 'AP')[0] >= C4_B for s in SEEDS))
    F5 = (all(is_pos(trip(s, 'B', 'AP')) and trip(s, 'B', 'AP')[0] >= C5_B for s in SEEDS)
          and all(trip(s, 'A', 'AP')[0] >= C5_A for s in SEEDS))
    routes = []
    if P['A']:
        routes.append(dict(gain='A', other='B', form='C4', T=T['B'], F=F4, out=('①a' if F4 else '①b') if T['B'] else '②'))
    if P['B']:
        routes.append(dict(gain='B', other='A', form='C5', T=T['A'], F=F5, out=('①a' if F5 else '①b') if T['A'] else '②'))
    if not routes:
        out = '③'
    else:
        outs = {r['out'] for r in routes}
        if len(outs) == 1:
            out = outs.pop()
        elif '②' in outs:
            out = '未覆盖'      # 两组都可重复为正，一条路径读成①、另一条读成②：§3.6 两行同时字面成立，另行裁决
        else:
            out = '①a'          # {①a, ①b}：①a 的条件字面成立，①b 要求两种形式都不成立
    return out, routes, dict(P=P, T=T, C4=F4, C5=F5)


def parse_gamma(gv):
    """所选 γ：字符串或数（不接受布尔），有限，且在第十七轮的 8 点网格内"""
    if isinstance(gv, bool) or not isinstance(gv, (str, int, float)):
        return None
    try:
        g = float(gv)
    except ValueError:
        return None
    if not math.isfinite(g):
        return None
    hit = [c for c in CONST_GRID if abs(c - g) <= 1e-12]
    return hit[0] if hit else None


def curve_check(rows, scope):
    """第十七轮 §3.6 第三条限制：逐种子比较 8 个已测 γ 的 QE+Const 与 QE+M（A、B 两组 mAP）。只限定措辞。"""
    res = dict(hits=[], near=[], problems=[], table={}, complete=True)
    for s in SEEDS:
        cv = rows.get(('r17', scope, s, 'curve'))
        qa, qb = as_dict(rows.get(('cells', scope, s, 'abs|QE+M|A'))), as_dict(rows.get(('cells', scope, s, 'abs|QE+M|B')))
        da, db = as_dict(rows.get(('cells', scope, s, 'abs|QE-DBA|A'))), as_dict(rows.get(('cells', scope, s, 'abs|QE-DBA|B')))
        qmA, qmB = num(qa.get('mAP')), num(qb.get('mAP'))
        if not isinstance(cv, dict) or qmA is None or qmB is None:
            res['problems'].append(f's{s} 曲线或 QE+M 的绝对 mAP 缺失')
            res['complete'] = False
            continue
        pts = {}
        for k, v in cv.items():
            g = parse_gamma(k)
            a, b = num(as_dict(v).get('A_mAP')), num(as_dict(v).get('B_mAP'))
            if g is None or a is None or b is None:
                res['problems'].append(f's{s} 曲线点不可读：{k!r}')
                res['complete'] = False
                continue
            if g in pts:
                res['problems'].append(f's{s} 曲线中 γ={g:g} 出现不止一次')
                res['complete'] = False
            pts[g] = (a, b)
        if sorted(pts) != list(CONST_GRID):
            res['problems'].append(f's{s} 曲线不是 8 个已测点：{sorted(pts)}')
            res['complete'] = False
        for g, (a, b) in sorted(pts.items()):
            if a > qmA and b > qmB:
                res['hits'].append((s, g))
            # 在 r17 对 r15 的复现容差内可能翻转"是否同时超过"的点：两组都不低于 QE+M − 1e-6，且至少一组相差不到 1e-6
            if a >= qmA - REPRO_TOL and b >= qmB - REPRO_TOL and (abs(a - qmA) <= REPRO_TOL or abs(b - qmB) <= REPRO_TOL):
                res['near'].append((s, g))
        res['table'][s] = dict(pts=pts, qm=(qmA, qmB), qd=(num(da.get('mAP')), num(db.get('mAP'))))
    return res


def curve_wording(checks):
    """第三条限制的写法：只有每个种子都核过 8 点、没有命中也没有需核对的点时，才用规定句式"""
    if any(c['hits'] for c in checks):
        return '有已测 γ 在某个种子上同时在两组超过 QE+M，必须如实写出是哪些 γ、哪些种子'
    if any(c['near'] or not c['complete'] for c in checks):
        return '第三条限制须人工核对（曲线不全，或有在复现容差内可能翻转的点；见各范围的提示），核对之前不写结论句'
    return '写"已测的 8 个常数取值都没有同时在两组上超过 QE+M"（不写"任何常数 γ 都达不到"）'


def analyze_scope(rows, scope):
    res = dict(scope=scope, missing=[], invalid=[], warnings=[], selection_false=False, seeds={}, tier=None)

    def R(src, seed, key):
        return rows.get((src, scope, seed, key))

    def D(src, seed, key):
        """取一行，不是字典就记为无效并返回空字典"""
        v = R(src, seed, key)
        if not isinstance(v, dict):
            res['invalid'].append(f's{seed} {src}:{key} 的内容不是字典：{v!r}'[:200])
            return {}
        return v

    if not any(k[1] == scope for k in rows):
        res['state'] = 'missing'
        return res
    # ---------- 完整性
    for s in SEEDS:
        need = [('status', 'judge'), ('status', 'r17'), ('status', 'selection_equal'),
                ('judge', 'pair|QE+M|QE-DBA|A'), ('judge', 'pair|QE+M|QE-DBA|B'), ('judge', 'claims'),
                ('cells', 'strongest|A'), ('cells', 'strongest|B'), ('cells', 'abs|QE+M|M0'), ('cells', 'abs|QE-DBA|M0'),
                ('r17', 'pair|QE+M-QE+Const|A'), ('r17', 'pair|QE+M-QE+Const|B'), ('r17', 'locality')]
        need += [('r17', f'gamma|f{f}') for f in FITS]
        need += [('status', f'runner|{p}/s{s}_f{f}') for p in RUNNER_PARTS[scope] for f in FITS]
        if scope == 'full':
            need += [('status', 'xcam'), ('xcam', 'pair|QE+M-xcam|QE-DBA|A'), ('xcam', 'pair|QE+M-xcam|QE-DBA|B')]
        for src, key in need:
            if R(src, s, key) is None:
                res['missing'].append(f's{s} {src}:{key}')
    if res['missing']:
        res['state'] = 'incomplete'
        return res

    def pair(s, x, y, g, required=True):
        """judge 配对差：dmAP、lo、hi、dR1、r_lo、r_hi、n_q、n_id。required=False 时缺失只返回 None。"""
        v = R('judge', s, f'pair|{x}|{y}|{g}')
        if not required and not isinstance(v, dict):
            return None
        v = D('judge', s, f'pair|{x}|{y}|{g}')
        out = {}
        bad = []
        for f in ('dmAP', 'lo', 'hi', 'dR1', 'r_lo', 'r_hi'):
            out[f] = num(v.get(f))
            if out[f] is None:
                bad.append(f)
                out[f] = NAN
        if bad:
            msg = f's{s} judge pair|{x}|{y}|{g} 的 {"、".join(bad)} 缺失或不是有限数'
            if required:
                res['invalid'].append(msg)
            else:
                res['warnings'].append(msg)
                return None
        out['n_q'], out['n_id'] = v.get('n_q'), v.get('n_id')
        return out

    for s in SEEDS:
        S = {}
        # ---------- 有效性（§3.8）
        js = D('status', s, 'judge')
        if not (js.get('evalset') == 'test' and is_int(js.get('exit_status')) and js.get('exit_status') == 0):
            res['invalid'].append(f's{s} 冻结 judge_test 的状态不对：{js}')
        r17 = D('status', s, 'r17')
        if not (r17.get('valid') is True and r17.get('reproduction_ok') is True and r17.get('problems') == []):
            res['invalid'].append(f's{s} 常数对照报告（r17_judge）不是 VALID 或复现不一致：{r17}')
        if scope == 'full':
            xs = D('status', s, 'xcam')
            if not (xs.get('status') == 'VALID' and xs.get('problems') == []):
                res['invalid'].append(f's{s} 跨相机报告不是 VALID：{xs}')
        if scope == 'subset':
            se = D('status', s, 'selection_equal')
            flag = se.get('SELECTION_EQUAL')
            if flag is False:
                res['selection_false'] = True
                res['invalid'].append(f's{s} SELECTION_EQUAL=false（差异 {se.get("differences")} 处）：先排查，查明之前不判档')
            elif flag is not True:
                res['invalid'].append(f's{s} 子集没有记录 SELECTION_EQUAL：{se}')
        expected = {f'runner|{p}/s{s}_f{f}' for p in RUNNER_PARTS[scope] for f in FITS}
        present = {k[3] for k in rows if k[0] == 'status' and k[1] == scope and k[2] == s and k[3].startswith('runner|')}
        if present != expected:
            res['invalid'].append(f's{s} 运行器单元与预期不符：多出 {sorted(present - expected)}')
        heads = {}
        for key in sorted(expected):
            v = D('status', s, key)
            ev, hd = v.get('events'), v.get('heads')
            if not (is_int(ev) and ev >= 0 and isinstance(hd, dict)
                    and all(is_int(c) and c >= 0 for c in hd.values()) and sum(hd.values()) == ev):
                res['invalid'].append(f's{s} 运行器记录格式不对：{key} {v}'[:200])
                continue
            bad = set(hd) - RUNNER_HEADS
            if bad:
                res['invalid'].append(f's{s} 运行器出现允许范围外的单类拟合：{sorted(bad)}（{key}）')
            for h, c in hd.items():
                heads[h] = heads.get(h, 0) + c
        S['runner_heads'] = heads
        q, d = D('cells', s, 'abs|QE+M|M0'), D('cells', s, 'abs|QE-DBA|M0')
        for k in ('mAP', 'R1'):
            a, b = num(q.get(k)), num(d.get(k))
            if a is None or b is None or abs(a - b) > EQ_TOL:
                res['invalid'].append(f's{s} 不变量不成立：M0 上 QE+M 与 QE-DBA 的 {k} 不同（{a} vs {b}）')
        loc = D('r17', s, 'locality')
        lack = [m for m in LOC_METHODS if m not in loc]
        if lack:
            res['invalid'].append(f's{s} 局部性记录缺：{lack}')
        for m, v in sorted(loc.items()):
            if not (isinstance(v, dict) and is_int(v.get('changed')) and v.get('changed') == 0):
                res['invalid'].append(f's{s} 不变量不成立：{m} 在前端 top-1 不是仅红外项时改变了 top-1（{v}）'[:200])
        # ---------- 读数
        pa, pb = pair(s, 'QE+M', 'QE-DBA', 'A'), pair(s, 'QE+M', 'QE-DBA', 'B')
        for g, p in (('A', pa), ('B', pb)):
            if not (is_int(p['n_id']) and p['n_id'] == N_ID[scope]):
                res['invalid'].append(f's{s} {g} 组的评价身份数为 {p["n_id"]}，应为 {N_ID[scope]}')
            nq = p['n_q']
            if not (is_int(nq) and (nq == N_Q_FULL if scope == 'full' else 0 < nq < N_Q_FULL)):
                res['invalid'].append(f's{s} {g} 组的评价查询数为 {nq}，'
                                      + (f'应为 {N_Q_FULL}' if scope == 'full' else f'应在 1 与 {N_Q_FULL - 1} 之间'))
        S.update(pa=pa, pb=pb)
        S['E1'] = bool(pa['dmAP'] >= E1_GAIN and pa['lo'] > 0)
        S['E2'] = bool(pb['dmAP'] >= TOL and pb['dR1'] >= TOL)
        S['claims'] = D('judge', s, 'claims')
        for e, c in (('E1', 'C1'), ('E2', 'C2')):
            if S['claims'].get(c) is not S[e]:
                res['invalid'].append(f's{s} {e} 的判读（{S[e]}）与冻结 judge_test 的 {c}（{S["claims"].get(c)}）不一致')
        # ---------- E3（只限定措辞：缺数据只记提示，不影响档次）
        zA, zB = as_dict(R('cells', s, 'strongest|A')).get('method'), as_dict(R('cells', s, 'strongest|B')).get('method')
        S.update(zA=zA, zB=zB, sa=None, sb=None, E3A=None, E3B=None)
        for g, z in (('A', zA), ('B', zB)):
            if z not in POOL:
                res['warnings'].append(f's{s} {g} 组最强通用后处理不在候选池内：{z!r}；E3 不可判')
                continue
            p = pair(s, 'QE+M', z, g, required=False)
            if p is None:
                res['warnings'].append(f's{s} 缺 judge 配对差 pair|QE+M|{z}|{g}；E3 不可判，参照 judge_test 的 C3')
                continue
            S['s' + g.lower()] = p
            S['E3' + g] = bool(p['dmAP'] >= E1_GAIN and p['lo'] > 0) if g == 'A' else bool(p['dmAP'] >= TOL)
        S['E3'] = (False if False in (S['E3A'], S['E3B']) else None if None in (S['E3A'], S['E3B']) else True)
        c3 = S['claims'].get('C3')
        if S['E3'] is not None and c3 is not S['E3']:
            res['warnings'].append(f's{s} E3 的判读（{S["E3"]}）与冻结 judge_test 的 C3（{c3}）不一致：'
                                   f'E3 只限定措辞，按规格文字核对最强通用后处理后再写')
        # ---------- E4 用到的常数对照
        trips = {}
        for g in 'AB':
            v = D('r17', s, f'pair|QE+M-QE+Const|{g}')
            for met in ('AP', 'R1', 'R5', 'R10', 'mINP'):
                t = v.get(met)
                t = tuple(num(x) for x in t[:3]) if isinstance(t, list) and len(t) >= 3 else (None, None, None)
                if None in t:
                    if met in ('AP', 'R1'):
                        res['invalid'].append(f's{s} 常数对照 QE+M−QE+Const {g} 组 {met} 缺失或不是有限数')
                    t = (NAN, NAN, NAN)
                trips[(g, met)] = t
        S['const'] = trips
        S['gamma'] = {}
        for f in FITS:
            gv = D('r17', s, f'gamma|f{f}').get('gamma')
            g = parse_gamma(gv)
            if g is None:
                res['invalid'].append(f's{s} f{f} 所选 γ 不在第十七轮的 8 点网格内或不可读：{gv!r}')
                g = NAN
            S['gamma'][f] = g
        if scope == 'full':
            xa = D('xcam', s, 'pair|QE+M-xcam|QE-DBA|A')
            S['xcam_A'] = tuple(num(xa.get(f)) for f in ('dmAP', 'lo', 'hi'))
            if None in S['xcam_A']:
                res['invalid'].append(f's{s} 跨相机 A 组配对差缺失或不是有限数')
                S['xcam_A'] = (NAN, NAN, NAN)
        res['seeds'][s] = S
    # ---------- 汇总
    sd = res['seeds']
    res['E1'] = all(sd[s]['E1'] for s in SEEDS)
    res['E2'] = all(sd[s]['E2'] for s in SEEDS)
    e3 = [sd[s]['E3'] for s in SEEDS]
    res['E3'] = False if False in e3 else (None if None in e3 else True)
    res['E1_all_nonpositive'] = all(sd[s]['pa']['dmAP'] <= 0 for s in SEEDS)
    res['E2_wording'] = dict(
        mAP=tol_wording([sd[s]['pb']['dmAP'] for s in SEEDS], [sd[s]['pb']['lo'] for s in SEEDS]),
        R1=tol_wording([sd[s]['pb']['dR1'] for s in SEEDS], [sd[s]['pb']['r_lo'] for s in SEEDS]))
    res['E2_nonnegative'] = all(sd[s]['pb']['dmAP'] >= 0 for s in SEEDS)

    def trip(s, g, met):
        return sd[s]['const'][(g, met)]
    res['E4'], res['E4_routes'], res['E4_trace'] = e4_outcome(trip)
    res['E4_other'] = {}
    for g in 'AB':
        w = {}
        for met in ('AP', 'R1'):
            pts, los = [trip(s, g, met)[0] for s in SEEDS], [trip(s, g, met)[1] for s in SEEDS]
            note = ''
            if all(p > 0 and l > 0 for p, l in zip(pts, los)):
                note = '（该组三个种子也都可重复为正）'
            elif all(p >= 0 for p in pts):
                note = '（该组三个种子点估计都不为负；"损害"一词按实际改写，不写"无损"）'
            w[met] = tol_wording(pts, los) + note
        res['E4_other'][g] = w
    res['E4_opposite'] = []
    for g in 'AB':
        for s in SEEDS:
            ap = trip(s, g, 'AP')[0]
            for met in ('R1', 'R5', 'R10', 'mINP'):
                x = trip(s, g, met)[0]
                if not (is_nan(ap) or is_nan(x)) and ap * x < 0:
                    res['E4_opposite'].append(f'{g} 组 s{s} {met} {sg(x)} 与 mAP {sg(ap)} 方向相反')
    res['gamma_zero'] = [f's{s} f{f}' for s in SEEDS for f in FITS if sd[s]['gamma'][f] == 0]
    res['curve'] = curve_check(rows, scope)
    for p in res['curve']['problems']:
        res['warnings'].append('常数曲线：' + p)
    if scope == 'full':
        res['xcam_positive'] = all(sd[s]['xcam_A'][1] > 0 for s in SEEDS)
    if res['invalid']:
        res['tier'] = '实现或协议问题'
    elif res['E1'] and res['E2']:
        res['tier'] = '有效'
    elif res['E1_all_nonpositive']:
        res['tier'] = '未见效果'
    else:
        res['tier'] = '有限适用'
    res['state'] = 'judged'
    return res


def combine(scopes):
    """两个范围的措辞取较保守者（规格 §3.7，L2 前补写）"""
    f, s = scopes['full'], scopes['subset']
    out = {}
    out['E2'] = {m: E2_ORDER[min(E2_ORDER.index(f['E2_wording'][m]), E2_ORDER.index(s['E2_wording'][m]))] for m in ('mAP', 'R1')}
    out['E3'] = False if False in (f['E3'], s['E3']) else (None if None in (f['E3'], s['E3']) else True)
    if '未覆盖' in (f['E4'], s['E4']):
        out['E4'] = '未覆盖'
    else:
        out['E4'] = E4_ORDER[min(E4_ORDER.index(f['E4']), E4_ORDER.index(s['E4']))]
    out['curve_hits'] = [(sc, x, g) for sc in SCOPES for x, g in scopes[sc]['curve']['hits']]
    out['E4_gain'] = {sc: sorted({rt['gain'] for rt in scopes[sc].get('E4_routes', [])}) for sc in SCOPES}
    out['differ'] = [k for k, a, b in (('E2 写法', f['E2_wording'], s['E2_wording']), ('E3', f['E3'], s['E3']),
                                       ('E4', f['E4'], s['E4']), ('E4 得益组', out['E4_gain']['full'], out['E4_gain']['subset']))
                     if a != b]
    return out


def analyze(text):
    P = parse(text)
    out = dict(parse=P, scopes={}, final=None, combined=None, reasons=list(P['problems']))
    if P['problems']:
        return out
    for sc in SCOPES:
        out['scopes'][sc] = analyze_scope(P['rows'], sc)
        r = out['scopes'][sc]
        if r['state'] == 'missing':
            out['reasons'].append(f'{SCOPE_ZH[sc]}没有任何 R18J 行（规格 §3.3 O3 要求全量与去重子集都判档）')
        elif r['state'] == 'incomplete':
            out['reasons'].append(f'{SCOPE_ZH[sc]}缺 {len(r["missing"])} 项：' + '；'.join(r['missing'][:12])
                                  + (' …' if len(r['missing']) > 12 else ''))
    if not out['reasons']:
        out['final'] = min((out['scopes'][sc]['tier'] for sc in SCOPES), key=TIER_ORDER.index)
        out['combined'] = combine(out['scopes'])
    return out


# ============================================================== 描述性表格（不参与判定）
def mean_cell(rows, src, scope, key, field, sd_on=True):
    v = rows.get((src, scope, 'mean', key))
    if not isinstance(v, dict) or not isinstance(v.get(field), dict):
        return '—'
    c = v[field]
    s = c.get('mean_2dp') if isinstance(c.get('mean_2dp'), str) else '—'
    if sd_on and isinstance(c.get('sd_2dp'), str):
        s += '±' + c['sd_2dp']
    if c.get('n') != len(SEEDS):
        s += f'(n={c.get("n")})'
    return s


def resolve(node, field):
    """按 r18_core 的字段名（字典键与列表下标以 "." 连接；键本身可能含小数点，如 γ = 0.05）取值"""
    parts, cur, i = field.split('.'), node, 0
    while i < len(parts):
        hit = None
        for j in range(len(parts), i, -1):
            k = '.'.join(parts[i:j])
            if isinstance(cur, dict) and k in cur:
                hit = (cur[k], j)
                break
            if isinstance(cur, list) and j == i + 1 and k.isdigit() and int(k) < len(cur):
                hit = (cur[int(k)], j)
                break
        if hit is None:
            return None
        cur, i = hit
    return num(cur)


def check_means(rows):
    """均值行与各种子未舍入值的纯 Python 复算是否逐位相同（两种加法次序任一相同即可）；只作提示，不参与判定"""
    bad, n = [], 0
    for (src, sc, seed, key), v in sorted(rows.items(), key=str):
        if seed != 'mean' or not isinstance(v, dict):
            continue
        for field, c in v.items():
            if not isinstance(c, dict) or 'mean' not in c:
                continue
            vals = [x for x in (resolve(rows.get((src, sc, s, key)), field) for s in SEEDS) if x is not None]
            n += 1
            if len(vals) != c.get('n') or not vals:
                bad.append(f'{src}|{sc}|{key}|{field}（n {c.get("n")} 对 {len(vals)}）')
                continue
            ok = False
            orders = [vals] if len(vals) < 3 else [vals, [vals[0], vals[1] + vals[2]]]
            for parts in orders:
                tot = 0.0
                for x in parts:
                    tot += x
                mu = tot / len(vals)
                if mu != c.get('mean'):
                    continue
                if len(vals) == 1:
                    ok = c.get('sd') is None
                    break
                dev = [(x - mu) * (x - mu) for x in vals]
                for dparts in ([dev] if len(dev) < 3 else [dev, [dev[0], dev[1] + dev[2]]]):
                    t2 = 0.0
                    for x in dparts:
                        t2 += x
                    if math.sqrt(t2 / (len(vals) - 1)) == c.get('sd'):
                        ok = True
                        break
                if ok:
                    break
            if not ok:
                bad.append(f'{src}|{sc}|{key}|{field}')
    return n, bad


def tables(out):
    rows = out['parse']['rows']
    L = []
    W = L.append
    abs_methods = ['B*', 'QE-DBA', 'QEir', 'KR', 'RVN', 'KRir', 'RVNir', 'GP-same', 'CAET', 'QE+CAET', 'QE+SRC2f', 'QE+EM',
                   'FeatImp-g', 'A-2f-ctxn', 'QE+M', 'QE+Orc', 'Oracle-A']
    W('\n######## 描述性表格（只报告，不参与判定；均值 ± 标准差由各种子未舍入值计算，ddof=1，ROUND_HALF_UP；'
      '表中的 -0.00 是绝对值不足 0.005 的负值，论文表格写 0.00）')
    for sc in SCOPES:
        if out['scopes'].get(sc, {}).get('state') != 'judged':
            continue
        tag = '（R6 去重子集）' if sc == 'subset' else ''
        if sc == 'subset':
            W('\n注（规格 §3.3 O3）：子集的可用性按子集图库重新抽取（同一分配种子），全量与子集的差异既来自删去身份，也来自重新抽取的可用性。')
        W(f'\n==== 表 R1{tag}：{SCOPE_ZH[sc]} 绝对指标（%），三种子均值±标准差')
        W('  注：A-2f-ctxn 即"只用迁移"。M0 两列：在 M0 上不生效、没有 M0 单元的方法，按 judge_test 的做法填 B★ 的值'
          '（表 R3 中显示为 =B*）。QE+Const、QE+NoGate 的绝对指标不在 R18J_core 中：未舍入值见全量 R18J.txt 中 '
          'src=r17、key=absolute.s<种子>:QE+Const.A/B（及 QE+NoGate）的行，或 report_const_<范围>.json 的 absolute，'
          '须按 r18_core 的做法求均值与标准差，不用 report_const 文本中的两位小数。')
        cols = ('A mAP', 'A R1', 'A R5', 'A R10', 'A mINP', 'B mAP', 'B R1', 'B mINP', 'M0 mAP', 'M0 R1')
        W(pad('方法', 11) + ' | ' + ' '.join(pad(c, 12, True) for c in cols[:5]) + ' | '
          + ' '.join(pad(c, 12, True) for c in cols[5:8]) + ' | ' + ' '.join(pad(c, 12, True) for c in cols[8:]))
        for m in abs_methods:
            if rows.get(('cells', sc, 'mean', f'abs|{m}|A')) is None:
                continue
            a = [mean_cell(rows, 'cells', sc, f'abs|{m}|A', k) for k in ('mAP', 'R1', 'R5', 'R10', 'mINP')]
            b = [mean_cell(rows, 'cells', sc, f'abs|{m}|B', k) for k in ('mAP', 'R1', 'mINP')]
            z = [mean_cell(rows, 'cells', sc, f'abs|{m}|M0', k) for k in ('mAP', 'R1')]
            W(pad(m, 11) + ' | ' + ' '.join(pad(x, 12, True) for x in a) + ' | ' + ' '.join(pad(x, 12, True) for x in b)
              + ' | ' + ' '.join(pad(x, 12, True) for x in z))
        W(f'\n==== 表 R2{tag}：{SCOPE_ZH[sc]} 配对差（百分点），每个种子一列，95% 身份簇 bootstrap 区间（4000 次，种子 7）')
        W(pad('比较', 30) + ' | ' + ' | '.join(center('s' + str(s), 34) for s in SEEDS))

        def jrow(label, x, y, g, met):
            cells = []
            for s in SEEDS:
                yy = y(s) if callable(y) else y
                v = rows.get(('judge', sc, s, f'pair|{x}|{yy}|{g}'))
                if not isinstance(v, dict) or v.get('missing'):
                    cells.append('—')
                    continue
                keys = ('dmAP', 'lo', 'hi') if met == 'mAP' else ('dR1', 'r_lo', 'r_hi')
                cells.append(ci(*(num(v.get(k)) for k in keys)) + (f' {yy}' if callable(y) else ''))
            W(pad(label, 30) + ' | ' + ' | '.join(center(c, 34) for c in cells))

        def strongest(g):
            return lambda s: as_dict(rows.get(('cells', sc, s, f'strongest|{g}'))).get('method')
        for g in 'AB':
            for met in ('mAP', 'R1'):
                jrow(f'QE+M − QE-DBA {g} {met}', 'QE+M', 'QE-DBA', g, met)
            jrow(f'QE+M − {g}组最强通用 {g} mAP', 'QE+M', strongest(g), g, 'mAP')
            for y in ('QE+CAET', 'QEir', 'QE+SRC2f', 'QE+EM', 'KRir', 'GP-same'):
                jrow(f'QE+M − {y} {g} mAP', 'QE+M', y, g, 'mAP')
            jrow(f'只用迁移 − FeatImp-g {g} mAP', 'A-2f-ctxn', 'FeatImp-g', g, 'mAP')
            jrow(f'QE+Orc − QE+M {g} mAP', 'QE+Orc', 'QE+M', g, 'mAP')
        for name in ('QE+M-QE+Const', 'QE+M-QE+NoGate', 'QE+Const-QE-DBA', 'QE+NoGate-QE-DBA'):
            for g in 'AB':
                for met in ('AP', 'R1'):
                    cells = []
                    for s in SEEDS:
                        t = as_dict(rows.get(('r17', sc, s, f'pair|{name}|{g}'))).get(met)
                        cells.append(ci(*(num(x) for x in t[:3])) if isinstance(t, list) and len(t) >= 3 else '—')
                    label = name.replace('-QE', ' − QE') + f' {g} {"mAP" if met == "AP" else met}'
                    W(pad(label, 30) + ' | ' + ' | '.join(center(c, 34) for c in cells))
        W(f'\n==== 表 R3{tag}：{SCOPE_ZH[sc]} 13 种混合逐一相对 B★ 的 dmAP（三种子均值；首行为 B★ 的 mAP；=B* 表示该方法在此混合上不生效）')
        W(pad('', 11) + ' ' + ' '.join(pad(x, 7, True) for x in MIXES))
        W(pad('B* mAP', 11) + ' ' + ' '.join(pad(mean_cell(rows, 'cells', sc, 'mixture_mAP|B*', x, False), 7, True) for x in MIXES))
        for m in ('QE-DBA', 'QEir', 'KRir', 'GP-same', 'FeatImp-g', 'CAET', 'A-2f-ctxn', 'QE+CAET', 'QE+EM', 'QE+M', 'Oracle-A'):
            if rows.get(('cells', sc, 'mean', f'mixture_dmAP|{m}')) is None:
                continue
            cells = []
            for x in MIXES:
                c = mean_cell(rows, 'cells', sc, f'mixture_dmAP|{m}', x, False)
                cells.append('=B*' if c == '—' and x == 'M0' else c)
            W(pad(m, 11) + ' ' + ' '.join(pad(c, 7, True) for c in cells))
        W(f'\n==== 表 R4{tag}：{SCOPE_ZH[sc]} 来源覆盖率 frac_src 与迁移质量 leak、coverage（三种子均值；— 为该混合上无定义）')
        dmix = [x for x in MIXES if x != 'M0']
        W(pad('', 20) + ' ' + ' '.join(pad(x, 7, True) for x in dmix))
        for m in ('QE+M', 'A-2f-ctxn', 'CAET', 'QE+CAET'):
            if rows.get(('diag', sc, 'mean', f'mixture_diag|{m}')) is None:
                continue
            for k in ('frac_src', 'leak', 'coverage'):
                W(pad(m + ' ' + k, 20) + ' ' + ' '.join(
                    pad(mean_cell(rows, 'diag', sc, f'mixture_diag|{m}', x + '.' + k, False), 7, True) for x in dmix))
        W(f'\n==== 常数对照：{SCOPE_ZH[sc]} 开发集所选 γ 与 QE+Const 已测网格的逐种子曲线（只报告；只代表已测的 8 个 γ）')
        for s in SEEDS:
            W(f'  s{s} 所选 γ：' + '，'.join(
                f'f{f} = {as_dict(rows.get(("r17", sc, s, f"gamma|f{f}"))).get("gamma")}' for f in FITS))
        cc = out['scopes'][sc]['curve']
        W('  ' + pad('种子 组 mAP', 14) + ' '.join(pad(f'γ={g:g}', 7, True) for g in CONST_GRID) + ' '
          + pad('[QE+M]', 8, True) + ' ' + pad('[QE-DBA]', 8, True))
        for s in SEEDS:
            t = cc['table'].get(s)
            if not t:
                continue
            for gi, g in enumerate('AB'):
                vals = [h2(t['pts'][x][gi]) if x in t['pts'] else '—' for x in CONST_GRID]
                ref = [h2(t['qm'][gi]), h2(t['qd'][gi]) if t['qd'][gi] is not None else '—']
                W('  ' + pad(f's{s} {g} 组', 14) + ' '.join(pad(v, 7, True) for v in vals) + ' '
                  + pad(ref[0], 8, True) + ' ' + pad(ref[1], 8, True))
    if out['scopes'].get('full', {}).get('state') == 'judged':
        W('\n==== 表 R5：跨相机子分析（只在全量上做；QE+M-xcam 只用其他相机的同身份完整项作供体）')
        W('  注：本表的 QE+M − QE-DBA 由跨相机运行内重算（与主评估的复现容差 1e-6），末位可能与表 R2 不同。')
        W(pad('比较', 28) + ' | ' + ' | '.join(center('s' + str(s), 30) for s in SEEDS))
        for x, y in (('QE+M-xcam', 'QE-DBA'), ('QE+M', 'QE-DBA'), ('QE+M', 'QE+M-xcam')):
            for g in 'AB':
                cells = []
                for s in SEEDS:
                    v = rows.get(('xcam', 'full', s, f'pair|{x}|{y}|{g}'))
                    cells.append(ci(*(num(v.get(k)) for k in ('dmAP', 'lo', 'hi'))) if isinstance(v, dict) else '—')
                W(pad(f'{x} − {y} {g}', 28) + ' | ' + ' | '.join(center(c, 30) for c in cells))
        cells = []
        for s in SEEDS:
            r = num(as_dict(rows.get(('xcam', 'full', s, 'ratio'))).get('ratio'))
            cells.append(h2(r) if r is not None else '—')
        W(pad('跨相机增益 / 全量增益', 28) + ' | ' + ' | '.join(center(c, 30) for c in cells))
        W('  来源覆盖（三种子均值；仅红外项中的比例）：任一相机有同身份完整项 / 有异相机的同身份完整项 / 只有同相机的同身份完整项')
        for mx in MIXES:
            if rows.get(('xcam', 'full', 'mean', f'coverage|{mx}')) is None:
                continue
            W(f'  {pad(mx, 6)} ' + ' / '.join(mean_cell(rows, 'xcam', 'full', f'coverage|{mx}', 'ratio.' + k, False)
                                              for k in ('same_identity_complete_any_camera', 'same_identity_complete_other_camera',
                                                        'same_identity_sources_only_same_camera')))
    n, bad = check_means(rows)
    W(f'\n[均值行核对] {n} 个字段按各种子未舍入值复算' + ('，全部逐位相同' if not bad else
                                                       f'，{len(bad)} 个不同（只作提示，不影响判定）：' + '；'.join(bad[:10])))
    return '\n'.join(L)


# ============================================================== 输出
def report(out, path, digest_sha):
    L = []
    W = L.append
    W(f'######## 论文4 第十八轮 判读（r18_verdict.py sha256 {sha256_file(os.path.abspath(__file__))}）')
    P = out['parse']
    W(f'输入：{path}（sha256 {digest_sha}）；R18J 行 {P["n_r18j"]}；'
      f'STATUS_L1={"/".join(P["status"]["L1"]) or "无"}；STATUS_L2={"/".join(P["status"]["L2"]) or "无"}')
    if P['problems']:
        W('\n[不判定] 输入不完整或不一致：')
        for p in P['problems']:
            W('  - ' + p)
        return '\n'.join(L)
    for sc in SCOPES:
        r = out['scopes'][sc]
        W(f'\n======== {SCOPE_ZH[sc]}')
        if r['state'] != 'judged':
            W('  [不判定] ' + ('没有任何 R18J 行' if r['state'] == 'missing' else '缺：' + '；'.join(r['missing'])))
            for p in r['invalid']:
                W('  ✗ ' + p)
            continue
        sd = r['seeds']
        W('[有效性]（规格 §3.8；任一不通过即"实现或协议问题"，先排查，不作方法结论）')
        if r['invalid']:
            for p in r['invalid']:
                W('  ✗ ' + p)
        else:
            W('  ✓ 全部通过：judge_test 状态；常数对照 VALID 且复现一致；'
              + ('跨相机报告 VALID；' if sc == 'full' else 'SELECTION_EQUAL=true；')
              + '运行器单元齐全且只有白名单内的单类拟合；M0 上 QE+M 与 QE-DBA 相同；局部性不变量；所选 γ 在网格内；'
                '评价身份数与查询数；E1、E2 与冻结 judge_test 的 C1、C2 一致')
        for p in r['warnings']:
            W('  ! ' + p)
        W('  运行器改为常数的单类拟合（只允许 anyh、srci、gateC）：' + '；'.join(
            f's{s} ' + ('，'.join(f'{h} {c} 次' for h, c in sorted(sd[s]['runner_heads'].items())) or '无') for s in SEEDS))
        W('[读法]（规格 §3.7；每条须在 3 个种子上都成立；阈值用未舍入值比较。显示两位小数，离门槛不到 0.005 时括号内附未舍入值）')
        W('  E1 A 组 QE+M − QE-DBA ≥ +1.00 且下界 > 0：' + '；'.join(
            f's{s} {cix(sd[s]["pa"]["dmAP"], sd[s]["pa"]["lo"], sd[s]["pa"]["hi"])} {yes(sd[s]["E1"])}' for s in SEEDS)
          + f' → {"成立" if r["E1"] else "不成立"}')
        W('  E2 B 组 QE+M − QE-DBA 的 mAP ≥ −0.30 且 R1 ≥ −0.30：' + '；'.join(
            f's{s} mAP {cix(sd[s]["pb"]["dmAP"], sd[s]["pb"]["lo"], sd[s]["pb"]["hi"])}，'
            f'R1 {cix(sd[s]["pb"]["dR1"], sd[s]["pb"]["r_lo"], sd[s]["pb"]["r_hi"])} {yes(sd[s]["E2"])}' for s in SEEDS)
          + f' → {"成立" if r["E2"] else "不成立"}')
        W(f'     E2 写法（第十七轮 §3.6 第一条限制，任何情况都不写"无损"或"可以忽略"）：mAP {r["E2_wording"]["mAP"]}；'
          f'R1 {r["E2_wording"]["R1"]}' + ('；三个种子 mAP 点估计都不为负，"代价"一词按实际改写' if r['E2_nonnegative'] else ''))
        parts = []
        for s in SEEDS:
            S = sd[s]
            a = (f'A 组最强 {S["zA"]} ' + (cix(S["sa"]["dmAP"], S["sa"]["lo"], S["sa"]["hi"]) if S['sa'] else '缺配对差')
                 + f' {yes(S["E3A"])}')
            b = f'B 组最强 {S["zB"]} ' + (sgx(S["sb"]["dmAP"]) if S['sb'] else '缺配对差') + f' {yes(S["E3B"])}'
            parts.append(f's{s} {a}，{b}')
        e3txt = '成立' if r['E3'] is True else ('不成立' if r['E3'] is False else '不可判（见上方提示）')
        W('  E3 对最强通用后处理（A：≥ +1.00 且下界 > 0；B：≥ −0.30）：' + '；'.join(parts) + f' → {e3txt}（只限定措辞）')
        tr = r['E4_trace']
        W(f'  E4 门控对常数（QE+M − QE+Const，第十七轮 §3.6）：结局 {r["E4"]}（只限定措辞）')
        for g in 'AB':
            W(f'     {g} 组 mAP：' + '；'.join(f's{s} {cix(*sd[s]["const"][(g, "AP")])}' for s in SEEDS)
              + '｜R1：' + '；'.join(f's{s} {cix(*sd[s]["const"][(g, "R1")])}' for s in SEEDS))
        W(f'     可重复为正（三个种子点估计都为正且区间不含 0）：A {yes(tr["P"]["A"])}，B {yes(tr["P"]["B"])}；'
          f'作另一组时每个种子 mAP、R1 ≥ −0.30：A {yes(tr["T"]["A"])}，B {yes(tr["T"]["B"])}；'
          f'C4 形式（A ≥ +0.50 且区间不含 0，B ≥ −0.10）{yes(tr["C4"])}；C5 形式（B ≥ +0.30 且区间不含 0，A ≥ −0.30）{yes(tr["C5"])}')
        for rt in r['E4_routes']:
            W(f'     路径：得益组 {rt["gain"]}，另一组 {rt["other"]}（{"满足" if rt["T"] else "不满足"} −0.30），'
              f'{rt["form"]} 形式{"成立" if rt["F"] else "不成立"} → {rt["out"]}')
        if not r['E4_routes']:
            W('     路径：两组都不可重复为正 → ③')
        W(f'     写法（第十七轮结局表）：{E4_WORDING[r["E4"]]}')
        if r['E4'] in ('②', '①b'):
            W(f'     第二条限制：{SECOND_LIMIT}')
        others = list(dict.fromkeys(rt['other'] for rt in r['E4_routes'])) or ['A', 'B']
        for g in others:
            W(f'     另一组 {g} 的写法（第一条限制）：mAP {r["E4_other"][g]["AP"]}；R1 {r["E4_other"][g]["R1"]}')
        if r['E4_opposite']:
            W('     与 mAP 方向相反、正文须写明：' + '；'.join(r['E4_opposite'])
              + '（注：§3.6 列出的是 R5、R10、mINP；R1 一并列出，供另一组写法与正文参考）')
        if r['gamma_zero']:
            W('     开发集选出 γ = 0 的拟合（正文写明"开发集规则在该次拟合中不选任何常数迁移"）：' + '，'.join(r['gamma_zero']))
        cc = r['curve']
        hits = '、'.join(f's{s} γ={g:g}' for s, g in cc['hits'])
        W('     第三条限制（常数曲线只代表已测的 8 个 γ）：同时在 A、B 两组 mAP 上超过同种子 QE+M 的已测 γ：' + (hits or '无')
          + ('；在复现容差 1e-6 内可能翻转、需核对：' + '、'.join(f's{s} γ={g:g}' for s, g in cc['near']) if cc['near'] else '')
          + ' → ' + curve_wording([cc]))
        if sc == 'full':
            W('  跨相机子分析（§3.9，A 组 QE+M-xcam − QE-DBA）：' + '；'.join(f's{s} {cix(*sd[s]["xcam_A"])}' for s in SEEDS)
              + ' → 写"' + ('跨相机供体仍带来正增益' if r['xcam_positive'] else '收益主要来自同相机供体') + '"，不写成跨视角泛化')
        W('  冻结 judge_test 的 claims（第十五轮形式，供对照）：' + '；'.join(
            f's{s} ' + ' '.join(f'{c}{"Y" if sd[s]["claims"].get(c) is True else "N"}' for c in ('C1', 'C2', 'C3', 'C4', 'C5'))
            for s in SEEDS))
        W(f'[档次] {r["tier"]}')
    W('\n======== 结论')
    if out['final'] is None:
        W('[不判定] ' + '；'.join(out['reasons']))
        return '\n'.join(L)
    f, s = out['scopes']['full'], out['scopes']['subset']
    W(f'档次：全量 {f["tier"]}；去重子集 {s["tier"]} → 论文取较保守的一档：{out["final"]}（由保守到宽松：' + '、'.join(TIER_ORDER) + '）')
    W(f'论文写法（规格 §8 第 1 条）：{TIER_SENTENCE[out["final"]]}')
    c = out['combined']
    if out['final'] != '实现或协议问题':
        W('措辞（全量与子集取较保守者，规格 §3.7）：')
        W(f'  - E2：mAP {c["E2"]["mAP"]}；R1 {c["E2"]["R1"]}；任何情况都不写"无损"或"可以忽略"')
        W(f'  - E3：' + ('成立' if c['E3'] is True else '不成立' if c['E3'] is False else '不可判（见上方提示）'))
        W(f'  - E4：{c["E4"]} → {E4_WORDING[c["E4"]]}'
          + ('（得益组：全量 ' + ('、'.join(c['E4_gain']['full']) or '无') + '；子集 ' + ('、'.join(c['E4_gain']['subset']) or '无')
             + '；"门控在 X 组"的 X 按得益组写，两个范围不同时分别写）' if c['E4_gain']['full'] or c['E4_gain']['subset'] else ''))
        if c['E4'] in ('②', '①b'):
            W(f'    {SECOND_LIMIT}')
        W('  - 常数曲线：' + curve_wording([f['curve'], s['curve']])
          + ('（' + '、'.join(f'{"全量" if sc == "full" else "子集"} s{x} γ={g:g}' for sc, x, g in c['curve_hits']) + '）'
             if c['curve_hits'] else ''))
        W('  - 跨相机子分析只在全量上做：' + ('跨相机供体仍带来正增益' if f['xcam_positive'] else '收益主要来自同相机供体')
          + '，不写成跨视角泛化')
        if c['differ']:
            W('  - 全量与子集不一致的项：' + '、'.join(c['differ']) + '（已按较保守者取；正文两者都报告）')
    W('数据集措辞（规格 §3.3 O4）：测试身份与 RGBT-PEDES 有重叠，写"第二数据集（与 RGBT-PEDES 有部分图像重叠，另报去重子集）"，不写"独立"。')
    return '\n'.join(L)


def main(argv):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except (AttributeError, ValueError):
        pass
    if len(argv) != 2:
        print(__doc__)
        return 1
    path = argv[1]
    try:
        with open(path, 'rb') as f:
            raw = f.read()
        text = raw.decode('utf-8')
    except (OSError, UnicodeDecodeError) as e:
        print(f'读取失败：{e}')
        return 1
    try:
        out = analyze(text)
        print(report(out, path, hashlib.sha256(raw).hexdigest()))
    except Exception:
        traceback.print_exc(file=sys.stdout)
        print('[脚本内部错误] 判读没有完成，请把以上输出原样发回')
        return 3
    if not out['parse']['problems']:
        try:
            print(tables(out))
        except Exception:
            traceback.print_exc(file=sys.stdout)
            print('[脚本内部错误] 判读已在上方完成；描述性表格没有打印完，请把以上输出原样发回')
            return 3
    return 0 if out['final'] is not None else 2


if __name__ == '__main__':
    sys.exit(main(sys.argv))
