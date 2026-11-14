# PRISM 12 小时研究计划（2026-11-13 23:30 起，CST）

> 执行人：Claude（本文件既是计划也是执行日志；每做完一步在 §5 记一行，包括 null 和作废的 run）。
> 结果汇总见 `REPORT.md`，文献笔记见 `lit/LIT_NOTES.md`，访谈方案见 `interview/PROTOCOL.md`。

## 0. 一句话

把项目从"用 LLM 给用户抽 persona / value profile，再做推荐"改成一个 **observational study**：
Bluesky 允许用户公开"点赞"一个 custom feed（数据集里叫 bookmark）。这是用户对"我想要哪种算法"的**显式表态**；
用户对 feed 内帖子的点赞则是**隐式（revealed）行为**。两者的关系正是原 proposal 的核心问题
（explicit profile 在 implicit behavior 之外有没有增量信息），只是换成了真实平台上可观察的选择，
并且可以用后续访谈交叉验证。

## 1. 为什么换方向（来自旧结果和文献）

| 旧尝试 | 结果 | 教训 |
|---|---|---|
| Bluesky 行为特征 KMeans persona（2,000 人） | 聚类主要反映活跃度 | 活跃度是最大混杂，任何对比都要先匹配活跃度 |
| LLM 78-value 打分（155 人） | Qwen 与 GPT 精确一致率 52%，聚类间差异模糊 | value 标注本身不稳，不能当主测量 |
| 正/弱负样本 embedding PCA | 弱负样本空间只反映 feed 话题 | "没互动"不等于"不喜欢"，曝光不可观测 |
| MIND explicit profile | 只有 implicit baseline（AUC 0.659），explicit 部分没做完 | 是 recsys 方法论文的路子，和 HCI/CSCW 目标不合 |
| Reddit newcomers（已是另一个项目） | 文本 alignment 全 null；network fit 预测留存；selection 而非 socialization | "显式/事前信号 vs 事后行为"和"selection vs influence"是可迁移的主线 |

查新（2026-11-13）：
- 两位老师组里已有两篇用**同一份数据**的论文：Pal et al. 2026《Algorithmic Cultivation》（feed 曝光 → 语言趋同，PSM + ATE）
  和 Pal et al. 2026《Hidden Toll of Social Media News》（News feed → 心理健康语言）。两篇都把"任何一次与 feed 的互动"当作 treatment，
  结果变量是语言。**所以本项目不能再做"feed → 语言"**。
- Bluesky feed 的其它工作研究的是 feed 创建者（Zhou, Wang & Zhang 2026，访谈 26 位 creator）和 feed 构建系统
  （Bonsai CHI'26、PILOT、Feed Elicitation、Paper Skygest）。**没有人研究订阅者一侧"说的"和"做的"之间的关系。**
- 理论抓手：stated vs revealed preference（Kleinberg et al. 2023；Milli et al. 2025；Milkman et al. 2009 "highbrow films gather dust"）、
  control 的安慰剂效应（Vaccaro et al. 2018）、selection vs influence（Aral 2009；Shalizi & Thomas 2011；Slater 2007 reinforcing spirals）、
  common identity vs bond（Ren, Kraut & Kiesler 2007，对应 Blacksky 这类身份型 feed 与 Science/News 这类话题型 feed）。

## 2. 数据

- 公开版 Bluesky Social Dataset（Failla & Rossetti 2024，Zenodo 14669616），与另一台服务器上的是同一份；已下载到
  `data/bluesky_zenodo/`（不进 git）。
- 显式信号：`feed_bookmarks.csv` = 用户点赞 feed generator，18,314 条 / 15,767 人 / 11 个 feed，带分钟级时间戳。
  注意它是**公开点赞**，不是私有的 pin/save（论文原文核对过）。
- 隐式信号：`feed_posts_likes` = 对 feed 所收录帖子的点赞，去重后 315 万条 / 25.2 万人。
  注意：点赞不代表是"通过 feed 看到的"，帖子也可能出现在 Following 时间线里。
- 只有 Science、Blacksky、News、#UkrainianView 四个 feed 的帖子存档覆盖了完整历史；其余 7 个只有最后几天，行为分析不用。
- `user_posts`（235M 帖，19.5 GB 压缩）用于活跃度、tenure、留存、文本。

## 3. 研究问题与事先写定的判据

写于看到 RQ1 描述统计和 treated-only 事件曲线之后、任何匹配/对照分析之前。

| # | 问题 | 分析 | 判据（事先定） |
|---|---|---|---|
| RQ1 | 显式表态和隐式行为重合多少？ | 2×2 重合表，分 feed | 描述性，无判据（已完成，exploratory） |
| RQ2 | 显式表态在隐式历史之外有没有增量信息？（原 proposal 的 hybrid 问题） | 预测 t0 后 2–8 周是否点赞该 feed 的帖子：implicit-only / explicit-only / hybrid / shuffled-explicit | hybrid − implicit 的 ΔAUC bootstrap 95% CI 不含 0，且 shuffled ≈ 0 |
| RQ3 | 表态之后行为变了吗，还是早就在变？（selection vs influence） | 按 t0 前的隐式互动轨迹 + 活跃度匹配对照，event-study DiD，周粒度 −8…+8 | (a) 匹配后 −8…−2 周 treated−control 差的 95% CI 含 0；(b) +2…+8 周 DiD > 0 且 CI 不含 0；(c) placebo outcome（其它 feed 的点赞）效应小于 target |
| RQ4 | 效应持续吗？ | +1…+8 周逐周估计 | +5…+8 周均值仍 > 0（CI 不含 0）记为"持续"，否则记为"新鲜感" |
| RQ5 | "只表态不互动"的人是谁？ | 用 user_posts 的发帖/回复/转发活跃度区分 lurker 与 aspirational endorser | 描述性 |
| RQ6 | 话题型 vs 身份型 feed 是否不同？ | RQ1–RQ4 分 feed 比较 | 描述性；每个 feed 单独报告，不合并后再挑 |
| RQ7 | 表态后自己写的内容是否向 feed 靠拢？（对 Pal et al. 的 event-study 复查，只做 Science） | 用户帖子 embedding 与 feed 质心的相似度，匹配对照，逐周 | 同 RQ3 的 (a)(b)；若 (a) 不成立，报告"趋同在表态前已开始" |
| RQ8 | 表态与平台留存 | t0 后 60–90 天是否仍活跃，匹配对照 | 风险差的 CI 不含 0 |

稳健性（每个因果性表述都要过）：placebo 日期、placebo feed、不同匹配规格、去掉 t0 当周、E-value、
多重比较用 BH-FDR、所有 null 都写进报告。

## 4. 时间表

| 时段 | 内容 | 产出 |
|---|---|---|
| 0–1.5 h | 读旧结果、proposal；查数据；下载公开数据集 | 本文件 §1–§2 |
| 1.5–3 h | 文献：两位老师 25+ 篇（本地语料库），外部 90 篇摘要，关键论文读全文 | `lit/LIT_NOTES.md` |
| 3–4 h | RQ1 描述统计；treated-only 事件曲线 | `results/rq1_*`, `results/rq2_event_profile_treated.csv` |
| 4–6 h | 解包 user_posts，建用户周面板（活跃度、tenure） | `data/derived/user_week.parquet` |
| 6–8 h | RQ3/RQ4 匹配 + event-study DiD + 稳健性；RQ2 预测 | `results/rq3_*`, `results/rq2_*`, 图 |
| 8–9.5 h | RQ5、RQ6、RQ8；RQ7（GPU 上算 embedding） | `results/rq5–8_*` |
| 9.5–11 h | 访谈方案（screener、提纲、与定量结果逐条对应的三角验证矩阵）；用 advisor-panel skill 过一遍两位老师会问的问题并补实验 | `interview/PROTOCOL.md`, `REPORT.md` §老师视角 |
| 11–12 h | 写 `REPORT.md`（含全部 null）、整理代码、提交到 GitHub 分支 | `REPORT.md` |

## 5. 执行日志（2026-11-13 23:21 – 11-14 约 02:30 CST）

| # | 做了什么 | 结果 | 产出 |
|---|---|---|---|
| 1 | 读 proposal、28 页更新记录、最早的 PRISM outline（中途补给） | 见 §1 | – |
| 2 | 查本地数据：只有 20 人；确认公开版数据集可下 | 下载 22 GB，MD5 通过；事后删掉没用到的 followers / graphs / interactions（2.4 GB） | `data/bluesky_zenodo/` |
| 3 | 文献 144 篇（两位老师 29 篇，外部 115 篇摘要，3 篇读了全文相关章节） | 发现组里已有两篇同数据论文；确认订阅者一侧是空白 | `lit/LIT_NOTES.md` |
| 4 | RQ1 重合度 | 说的和做的重合很少 | `02`, `results/rq1_gap_table.csv` |
| 5 | 流式解析 2.33 亿帖 → 用户×周面板 | 236 万发帖用户；发现 Blacksky 存档只有 2024-02/03，排除 | `04`, `05` |
| 6 | RQ3/RQ4/RQ8 匹配 event study，4 个规格 | 判据 (a)(b)(c) 满足；行为先于表态；持续但衰减 | `06`, `09`, `results/rq3_*`, 图 1 |
| 7 | RQ5 背书者类型 | 30–43% 活跃但零互动 | `08` |
| 8 | RQ2 预测 | 历史丰富时增量极小（News 不满足判据）；冷启动 O/E 1.8–6.6（事后分析） | `07`, `07b` |
| 9 | Line B：LLM 推断 profile 排序 | explicit < implicit，hybrid 最优权重 0，value≈随机 | `10`–`15`, `results/rankB_*` |
| 10 | Line C：合成真值用户 | 假设成立时推断 profile 有用，保真度预测效用 | `16`, `results/synC_*` |
| 11 | RQ7 写作趋同（Science） | 背书前已有差距，背书后无增加 | `17`, `18` |
| 12 | 访谈方案、总报告 | – | `interview/PROTOCOL.md`, `REPORT.md` |

作废 / 出错的 run（照实记）：
- `07_rq2_prediction.py` 第一次运行在 Blacksky 人群 B 上崩溃（只有一类标签），浪费约 40 分钟；Science 的两行从日志恢复，News / #UkrainianView 重跑。
- `11_rank_build.py` 第一版负样本取"点赞前 48 小时内的帖子"，recency 单独就有 AUC 0.82；改为"与正样本发帖时间 ±6 小时"后 recency 仍有 0.73，第一版结果作废。
- `06` 加 active 规格时改坏了一行常量，重跑一次。
- 第一版报告之后补做了 placebo 日期检验（背书周提前 16 周）和 E-value。**placebo 没通过**（Science +14.5，与真实 +14.2 一样大）：RQ3 判据 (b) 虽然字面满足，但不能解读为背书的效应；`REPORT.md` 的结论已改写为 selection。
- 没做：FDR、换匹配粒度、advisor-panel 推演、约 15 篇 personalization 论文。原因是优先做了用户中途追加的两条旧 proposal 线。

原计划 12 小时，实际约 3 小时把计划内的实验跑完（数据下载和解析比预计快得多）。剩下的工作见 `REPORT.md` §5，都需要新数据、IRB 或你的决定。

## 6. 第二轮：补稳健性（2026-11-14 上午，判据写于运行之前）

用户确认主线写成"说的和做的不重合 + selection"；IRB、新数据采集、预注册先不做，等教授意见。本轮只补第一轮列为"没做"的项目。
下面的设计和判据在写任何新代码、看任何新结果之前定下；本节先单独提交到 git，再运行。

### 6.1 同一批人上的真实日期 vs placebo 日期

- 主 placebo 位移 S = 13 周：placebo 的结果窗口（+2…+8）对应真实事件时间 −11…−5，不碰背书前最后 4 周。S = 16、20 作为敏感性。
  （第一轮用的是 S = 16，样本 199 人；改用 13 是为了扩大同批样本，这个选择在看结果之前做出。）
- "同批样本" = 真实日期分析里匹配上的背书者中，同时满足 `first_seen ≤ c−S−1` 且 placebo 日期前至少有 4 周 feed 存档的人；
  真实与 placebo 都只在这批人上估计，并报告差值（真实 − placebo），三者用同一组 bootstrap 权重。
- Bootstrap 改为按用户的 Bayesian bootstrap（treated 与对照用户都带权重），这样对照被重复使用时 CI 不会偏窄。

### 6.2 按轨迹构造对照的四个设计

| 代号 | 对照 | 匹配 / 调整内容 |
|---|---|---|
| T1 seq8 | 从未背书者 | 背书前 8 周"每周是否点赞该 feed 帖子"的**完整 0/1 序列**精确匹配（256 种）+ 8 周点赞总量、活跃度、资历、其它 feed 点赞、语言 |
| T2 persist16 | 从未背书者 | 前 16 周里有点赞的周数（持续性）与总量 + 最近一周与前 3 周的点赞数 + 活跃度、资历、其它 feed、语言 |
| T3 nyt | **尚未背书者**：之后 ≥13 周才背书同一 feed 的人 | 活跃度、近 4 周点赞数、近 8 周点赞周数、资历、语言 |
| T4 regadj | 从未背书者（训练结果模型） | 梯度提升模型用前 16 周逐周点赞数等预测 +2…+8 周互动率；对背书者算 观测 − 预测 |

判据（每个设计 × 每个 feed）：
- **placebo 通过** = 同批样本上 S=13 的 placebo DiD 的 95% CI 完全落在 ±3 个百分点内。±3 是事先的主观取值（第一轮最小的真实差距是 7 点，取其一半以下）。
  CI 含 0 但宽于 ±3 记为"不确定"；CI 不含 0 且点估计超出 ±3 记为"不通过"。
- 只有 placebo 通过的设计，其真实日期 DiD 才可以读作"背书之后的变化"；并且要求 真实 − placebo 的 CI 不含 0。
- placebo 不通过的设计，只报告 真实 − placebo（依赖"选择偏差不随事件时间变化"的假设，并写明）。
- 所有设计、所有 feed、所有位移都报告，不挑。

### 6.3 匹配粒度敏感性（对第一轮的 full 规格）

粗化分箱更粗 / 更细各一版；每人对照数 K = 1、5、10；匹配截止时点 L = 1、3、5、9、13 周前（看差距从什么时候开始拉开）。
判据：若 DiD 在这些变体间的变化小于其自身 CI 宽度，记为"对粒度不敏感"；否则照实报告范围。这是描述性的，不改变 6.2 的结论规则。

### 6.4 多重比较

对报告里用到的全部检验统一做 Benjamini–Hochberg（q = 0.05），分族：RQ3（feed × 规格 × 统计量 × 结果变量）、RQ2、RQ7、Line B、本轮新增。
p 值由 bootstrap 标准误的正态近似得到。报告校正前后显著的个数，并逐条列出因校正而不再显著、且报告正文引用过的结果。

### 6.5 其它

- advisor-panel 推演两位老师会问的问题；由此追加的分析各自在运行前把判据补记在本节之后。
- 补读第一轮没读到的论文（清单见 `lit/LIT_NOTES.md` J 末尾），Mem-PAL 与 BELA 读全文。

### 6.6 看到 6.1–6.3 的结果之后追加的检验（post hoc，如实标注；写于运行之前）

6.1–6.3 的结果（见 `results/rq3r_*`）暴露了两个设计问题，下面的检验是为它们加的，不是事先计划的：

1. **placebo 的幸存者偏差。** placebo 日期上的"伪背书者"在 S 周之后才背书，所以在整个 placebo 窗口里必然还在平台上；从未背书的对照没有这个保证。
   这会抬高 placebo 差距，使"placebo ≈ 真实"的结论偏向 selection。检验：placebo 里只用"在 a+S 周或之后仍有任何痕迹"的对照重做 full 和 seq8。
   判读：若 placebo DiD 因此落入 ±3 点，第一轮"placebo 没通过"主要是幸存者偏差，selection 的结论要削弱；若仍在 ±3 之外，结论保留。
   同理，T3（尚未背书者当对照）在**真实**日期上的偏差方向相反（对照必然活到之后），会压低估计。检验：treated 也只留"c+13 周或之后仍有痕迹"的人；
   并把间隔从 13 周放宽到 17、21 周（对照在结果窗口里离自己的背书更远）。
2. **基线水平不可比。** News 的存档量在第 37 周前后涨了约 5 倍，placebo 窗口落在低量期，百分点差不可比。
   追加比值指标：背书者与对照在 +2…+8 周的互动率之比（RR）及其 真实/placebo 之比。判读：只描述，不设阈值。
3. **"行为先于表态"是否只是这类人一贯的上升趋势。** 用尚未背书者当对照、只匹配到 −5 周（和 −9 周），看背书前 4 周是否仍高于对照。
   判读：−4…−1 周差距的 CI 不含 0 记为"背书前确有额外上升"；否则第一轮的"行为先于表态"应改述为"这类人的互动一直在上升"。

### 6.7 advisor-panel 之后追加的分析（A1–A8；推演见 `advisor/PANEL.md`；判读规则写于运行之前）

这些分析针对两位老师最可能问的问题，全部只用现有公开数据。都是 exploratory，结论只作描述，不做因果表述。

| # | 分析 | 判读规则（事先定） |
|---|---|---|
| A1 | 以背书时刻为零点，按小时统计对该 feed 帖子的点赞；对照 = 同一用户 ±7、±14、±21 天的同一时刻；负对照 = 同一窗口内对其它 feed 帖子的点赞 | ±1 小时内的点赞率 ≥ 基线的 5 倍，且该倍数高于其它 feed 的倍数，记为"背书发生在正在消费该 feed 内容的会话里"；否则记为"背书与消费在时间上不绑定" |
| A2 | 重新下载关注关系（快照，无时间戳），把对 feed 帖子的点赞拆成"关注了作者 / 没关注作者"，分别做事件研究（尚未背书者对照） | 背书当周的相对跳升（RR）在"没关注作者"一类上更大，记为"背书伴随经由 feed 的消费"；两类相同，记为"只是对话题的兴趣整体上升"。快照里背书之后才关注的作者会被算成"已关注"，所以"没关注"一类是下界，写明 |
| A3 | 背书者中自己的帖子被该 feed 收录的比例；贡献者与纯受众的背书率 | 描述性 |
| A4 | 11 个 feed：背书相对账号首次可见痕迹的时间；同一小时 / 同一天内背书多个 feed 的比例；无先前痕迹的背书者之后 8 周的行为；feed 内请求用户点赞该 feed 的帖子 | 描述性。"无先前痕迹"不等于新用户（可能是一直潜水的人），报告时不写成"新用户" |
| A5 | 事件研究按背书前 8 周有无互动、按资历分层 | 某一层 +5…+8 周的差距 CI 不含 0 且 treated 水平高于其自身 −8…−2 周水平，记为"起点"；回到对照水平，记为"高点" |
| A6 | 背书者内部：以"之后 8 周是否点赞该 feed 的帖子"为因变量的 logistic 回归；每周 feed 内点赞中背书者的份额 | 描述性，报告 OR 和 CI |
| A7 | full 与 nyt 两个设计匹配前后的标准化均值差（SMD） | 匹配后 |SMD| < 0.2 记为平衡（两位老师论文里用过 0.15–0.3） |
| A8 | 在 Pal et al. 的 treatment 定义下（首次任何互动为锚），Science：写作相似度的逐周曲线、pre-trend、placebo 日期 | 匹配后 −8…−2 周 treated−control 的趋势斜率 CI 含 0 且 placebo 日期的 DiD CI 含 0，记为"在他们的定义下看不到事前趋势"，并明确写 RQ7 不构成对该文的质疑；否则照实报告，并注明样本、编码器、匹配都与原文不同 |
