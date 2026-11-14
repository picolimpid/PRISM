# Advisor panel 推演：Koustuv + Eshwar 会怎么看这个项目（2026-11-14）

> 这是根据两位老师公开发表的论文和文章做的模拟，不是他们本人；真正的意见以组会为准。
> 标注方式：`[quote <id> p.N]` 是论文原文；`[inferred: ids]` 是从几篇论文归纳的做法；`[Claude]` 是我自己的建议，在他们的文字里找不到依据。
> 论文 id 可以用 `koustuv-skill-distill/scripts/search.py --paper <id>` 查到。

## 项目复述

用公开的 Bluesky 数据集（Failla & Rossetti 2024；Science、News、#UkrainianView 三个 feed；2023-06 至 2024-03）做观察研究：
用户公开点赞一个 custom feed 是「说」，点赞该 feed 收录的帖子是「做」。主线是两者重合很少，并且背书之后的互动差距来自 selection；
之后用访谈交叉验证。目标 CSCW / CHI。IRB、访谈和新数据采集要等两位老师同意。

## 1. 他们都会认可的

1. **观察 + 访谈的结构。** 这是 Koustuv 组现在的标准做法，CHI 2026 的 AI companion 论文就是准实验加访谈再加理论视角 [inferred: yuan2026mental, yoo2025ai]。
2. **自己做了 placebo 并照实报告没通过。** placebo 日期是他们的常规检验："We permute (randomize) several placebo dates within the pre-enrollment data." [quote saha2024observer p.13]
3. **问题落在「用户对 feed 的控制」上。** r/popular 审计的结尾写的就是这个问题："should users have control over how they interact with content and how their content is distributed online?" [quote chan2026examining p.11]

## 2. 两人共同的标准，用到这个项目上

- 他们合写的报告清单要求把三类结论分开："Separate prediction, association, and causal interpretation." [quote saha2026from p.9]
  本项目现在的结论全部是描述性或预测性的，没有一条是因果的；论文里每条发现都要标明类型（见第 6 节表格）。
- "A trace can support more than one interpretation." [quote saha2026from p.5]
  「点赞 feed」至少有三种读法（订阅、支持创建者、公开表态），「点赞 feed 里的帖子」至少两种（通过 feed 看到、在 Following 里看到）。这是全文最大的风险。
- "Matching addresses measured differences, while unmeasured confounding can remain" [quote saha2026from p.6]
  第二轮的 placebo 结果正是这句话的实例：对从未背书者怎么匹配，Science 上都留下约 14 个百分点的差距，而这个差距在背书前 13–20 周就有。

## 3. Koustuv 会追问的

1. **这个痕迹到底量的是什么？而且我们组的论文把它当成了别的东西。**
   News 那篇把 bookmark 算作曝光的一种："exposure could be in the form of bookmarking, liking, commenting, reposting, quoting, or posting on this feed" [quote pal2026hidden p.4]；
   他在 op-ed 里把它解释成被动消费："Passive behaviors—such as bookmarking multiple news feeds or silently consuming headlines—were associated with substantially worse emotional outcomes than more active forms of engagement like commenting or contextualizing news through discussion." [quote web:koustuv_oped_6]
   本项目说它是「表态」，并且发现 News 的背书者里 46% 从未点赞过该 feed 的任何帖子。他会问：*凭什么说这是表态而不是静默消费？没点赞的人是没看，还是看了不点？* [inferred: saha2026from, saha2021dissertation]
   日志答不了后半句（访谈 Q4–Q7 才能答），但日志可以答前半句的一部分，见第 7 节 A1–A4。
2. **数据里是谁，结论想推到谁身上？** [inferred: saha2021dissertation, saha2019social]
   21–29% 的背书者一到平台就点了 feed，被排除在事件研究之外；时段是邀请制到刚开放；只有三个 feed。需要一张「纳入 / 排除」流程表，并单独描述被排除的人。
3. **随时间怎么变，对谁不一样？** 他不接受单一结论，会要异质性和时间维度 [inferred: yuan2026mental, pal2026hidden, web:koustuv_oped_6]。
   具体会问：*对没有历史的人，背书是不是互动的起点；对已经在互动的人，是不是只是高点？* 见第 7 节 A5。
4. **理论是什么？** 他习惯先从心理学或传播学借一个理论再去操作化 [inferred: yuan2026mental, saha2024observer]。
   stated vs revealed preference 是经济学的说法；他可能更想看到心理学一侧的框架，例如 intention–behavior gap、自我呈现（他自己用过 observer effect 和 self-presentation）[Claude]。这部分文献第一轮没有读，需要补。
5. **和 Pal et al. 的关系。** 那篇自己写了局限："Although our quasi-experimental design mitigates confounders compared to purely correlational approaches, it does not establish true causality." [quote pal2026algorithmic p.10]
   本项目的 RQ7（Science 背书者的写作在背书前就更接近 feed，背书后没有增加）会被读成对它的质疑。他不会反对做检验，但会要求在**他们的 treatment 定义下**（首次任何互动）做同样的 pre-trend 和 placebo，而不是换一个锚点就下结论 [Claude]。见第 7 节 A8。

## 4. Eshwar 会追问的

1. **三个 feed 是三种东西，不要合并。** 他的基本立场是价值和规范因社区而异："value is fundamentally contextual" [quote goyal2026vastu p.8]。
   Science / News 是话题型，#UkrainianView 是立场型，身份型的 Blacksky 因为存档问题缺席。他会问：*单位是用户还是 feed？11 个 feed 都有背书记录，为什么只讲 3 个？* 见第 7 节 A4。
2. **谁会拿这个结果去做事？** 他评估干预时总是对着平台或社区自己的目标来问 [inferred: chandrasekharan2022quarantined, chandrasekharan2017you]，并且主张把 feed 的控制权交给社区："we also suggest that platforms give communities more agency over their feeds." [quote lambert2024positive p.23]
   对 feed 创建者，可操作的结论是「点赞数不等于受众」；对平台，是 feed 发现页用点赞数排序是否合理。需要一张创建者视角的表：每周 feed 内互动里有多少来自背书者。见第 7 节 A6。
3. **把它当成对一个平台指标的审计。** 他组里第三个方向就是审计决定人们看到什么的系统 [inferred: chan2026examining, huang2026answer]。
   Bluesky 在 feed 列表里公开显示「被多少人点赞」，这是平台给 feed 的公开受欢迎度指标；本项目可以定位成对这个指标的审计 [Claude]。
4. **曝光量不到，怎么写？** 他们自己遇到过并且直接写进局限："our measure of exposure to explanations is limited by our data access and is not precise." [quote jhaver2024bystanders p.7]
   他会接受这个局限，但要看到尽力而为的代理指标（会话级时间、是否关注作者）。见第 7 节 A1、A2。
5. **「说的」和「实际的」不一致，他组里有先例。** r/ChangeMyView 的问卷发现用户对规则的支持高、对执行的支持低 [inferred: koshy2023measuring]；
   Language of Approval 的结论里写到 "gap between what is taught and explicitly stated in community guidelines, and what works in practice" [quote goyal2026language p.20]。可以把本项目接到这条线上。

## 5. 两人可能侧重不同的地方（开放问题，不是他们之间的分歧）

- **单位**：Koustuv 看个人轨迹和心理机制（为什么说了不做）；Eshwar 看 feed / 社区层面和平台设计。论文的主分析单位需要选一个。
- **怎么处理与 Pal et al. 的张力**：作为那篇的稳健性补充、作为本文的一节，还是完全不提。这需要他们决定，也要先和一作沟通。
- **投哪里**：审计 + 设计含义偏 CSCW；加入心理学理论和访谈偏 CHI；纯观察偏 ICWSM。

## 6. 按他们的报告清单自查（saha2026from p.9 的八项）

| 清单项 | 现状 | 缺什么 |
|---|---|---|
| Target population | 三个 feed、时段、语言已写 | 纳入 / 排除流程表；一到平台就背书的人单独描述（A4） |
| Question and estimand | RQ1–RQ8 有判据 | 每条发现标明是描述、预测还是因果；现在没有因果结论 |
| Constructs and measures | 核对过 bookmark 是公开点赞而不是 pin | 「说」和「做」两个构念的效度证据（A1–A3），访谈之前只能做到部分 |
| Collection and provenance | 公开数据集，MD5 校验；Blacksky 存档只有 2024-02/03 | 写明存档覆盖的检查方法；这个问题对组里另两篇也有影响，要当面讲 |
| Assignment and timing | 按日历周分 cohort；事件研究 −8…+8 周 | 小时级的时间结构（A1） |
| Adjustment and overlap | 粗化精确匹配，报告了匹配率 | 匹配前后的 SMD 平衡表（A7）；匹配丢掉了谁 |
| Sensitivity and uncertainty | placebo 日期、四种对照设计、粒度、E-value、bootstrap | 多重比较校正（A9）；异质性（A5） |
| Interpretation and reuse | 代码和结果在仓库 | 把 selection 的结论和「背书是信息不是干预」写成正文的解释框架 |

## 7. 建议的下一步：按对论文的威胁排序，每条对应一个只用现有数据就能做的分析

| # | 他们会问的问题 | 现有数据上能做的分析 | 状态 |
|---|---|---|---|
| A1 | 点赞 feed 的那一刻，用户是不是真的在用这个 feed？ | 以背书时刻为零点，按小时统计对该 feed 帖子的点赞；与同一用户其它日子的同一时段比较 | 待跑 |
| A2 | 对 feed 内帖子的点赞，是通过 feed 看到的，还是在 Following 里看到的？ | 重新下载关注关系，把点赞拆成「关注了作者」和「没关注作者」两类，看背书前后的变化集中在哪一类 | 待跑 |
| A3 | 背书者是受众还是内容贡献者？ | 背书者中有多少人自己的帖子被 feed 收录；贡献者与纯受众的背书率 | 待跑 |
| A4 | 11 个 feed 的背书行为是什么样？一到平台就背书的人是谁？ | 背书相对账号首次出现的时间；同一小时内连续背书多个 feed 的比例；到达即背书者之后 8 周的行为 | 待跑 |
| A5 | 对谁是起点，对谁是高点？ | 事件研究按背书前有无互动历史、按资历分层 | 待跑 |
| A6 | 谁说了就做，谁说了不做？创建者看到的受众是谁？ | 以「背书后 8 周是否互动」为因变量的回归；每周 feed 内点赞中背书者的份额 | 待跑 |
| A7 | 匹配后两组到底像不像？ | 匹配前后各协变量的标准化均值差 | 待跑 |
| A8 | 在 Pal et al. 的 treatment 定义下，pre-trend 和 placebo 是什么样？ | 以首次任何互动为锚，画写作相似度的逐周曲线并做 placebo 日期 | 待跑 |
| A9 | 这么多检验，校正之后还剩什么？ | 全部检验做 BH-FDR | 待跑 |

## 8. 他们组里最接近的工作，以及可以借的做法

- pal2026algorithmic、pal2026hidden：同一数据集。借：给对照分配 placebo 日期并用 KS 检验核对分布；分层倾向得分。本项目补的是 pre-trend、placebo 日期和「背书」与「互动」的拆分。
- yuan2026mental：准实验 + 访谈 + 理论视角的整体结构；parallel trends 的验证放附录。
- saha2024observer：150 个随机 placebo 日期的置换检验。本项目现在只有 3 个固定位移，可以照它扩成一个分布。
- lambert2025positive、chandrasekharan2017you：社区内分层匹配 + DiD，分用户类型报告。
- chan2026examining、chan2024understanding：把曝光系统当作审计对象的写法。
- koshy2023measuring、goyal2026language：「说的」与「实际的」不一致的先例。
- jhaver2024bystanders：曝光测不准时局限怎么写。

## 9. 带去真实组会的问题

1. 论文的主张定在「说的和做的不重合，背书是信息而不是干预」，是否同意？因果部分只作为反面结果（selection）出现。
2. RQ7 和第二轮的 placebo 结果与 Pal et al. 的关系怎么处理？是否需要先和 Olivia 对一下，在她的设定下重跑？
3. 我们的公开版数据里 Blacksky 的帖子存档只覆盖 2024 年 2–3 月。组里那份是不是一样？如果一样，Blacksky 上「首次互动」的日期会集中在这两个月。
4. 主分析单位选用户还是 feed？
5. 访谈招募范围：只招这三个 feed 的用户，还是任何使用 custom feed 的人？访谈里给受访者看根据其公开点赞生成的 profile，IRB 上是否可行？
6. 是否值得从 AT Protocol 采 2025–2026 年几百个 feed 的新数据做预注册复现？这能解决「只有三个 feed」和「早期用户」两个问题。
7. 投稿目标和时间：CSCW、CHI 还是 ICWSM？

## 证据（本次检索用到的 id）

saha2026from, pal2026algorithmic, pal2026hidden, yuan2026mental, saha2024observer, saha2025mentalb, jhaver2024bystanders, lambert2025positive,
lambert2024positive, chan2026examining, chan2024understanding, goyal2026language, goyal2026vastu, koshy2023measuring, chandrasekharan2017you,
chandrasekharan2022quarantined, saha2019social, saha2021dissertation, yoo2025ai, huang2026answer, web:koustuv_oped_6。
检索共 9 次查询（两位老师各自的语料库和合并语料库），并读了上面带页码的各页原文。
