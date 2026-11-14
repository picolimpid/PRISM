# 文献笔记（2026-11-13，Claude 逐篇阅读）

**阅读方式（如实说明）**
- 两位老师的论文：从本地语料库 `koustuv-skill-distill`（224 篇）里选了 29 篇，读了摘要和方法/稳健性笔记；
  其中 Pal et al. 2026《Algorithmic Cultivation》读了全文第 1–6 页（引言、相关工作、数据、匹配方法、结果变量定义）。
- 外部论文：85 篇读了完整摘要（OpenAlex 抓取，另有 5 条检索命中了错误论文，已剔除，不计入）；
  4 篇 arXiv 预印本读了摘要（Feed Elicitation、Paper Skygest、Middleware in Practice、PILOT）。
- 读了全文相关章节的：Failla & Rossetti 2024（数据采集与 feed 部分）、Greenwood & Garg 2026 Paper Skygest（采纳前后分析）、
  Zhou, Wang & Zhang 2026（方法、subscriber 定义、局限）。
- 没有用任何模型筛选或总结论文；检索用的是标题精确查询和 3 次网页搜索（查新）。
- 第二批（用户中途要求补读 personalization / 合成数据方向）：26 篇摘要，见 §3 和清单 J。其中 OpenAlex 当晚限流，另有约 15 篇计划内的论文没抓到，列在 J 末尾"未读"。
- 合计 144 篇（第一轮）。
- **第二轮（2026-11-14 上午）读了 12 篇摘要（10 篇是新的）+ 2 篇全文**，见 §4 和清单 K：
  第一轮列为"未读"的 8 项里读到了 6 项的摘要（Park et al. 2024、Argyle et al. 2022、Santurkar et al. 2023、Social Simulacra、TEARS、Epstein et al. 2025），
  另读了计划清单里第一轮没抓到的 6 篇（Xie et al. 2024、PersonaChat、PersonaMem、Alexandria 的 arXiv 版摘要、Jahanbakhsh et al. 2025、MIND）；
  **BELA 读了正文全部六节（不含附录），Mem-PAL 读了正文全部（不含附录）**。
  仍然没读到：STELA（arXiv 上没有，OpenAlex 和 Semantic Scholar 当天都限流）、"Enabling User Agency in Scalable Content Recommendations"（同上）、
  Li et al. AAMAS 2026（没有标题，查不到）。intention–behavior gap 一类心理学文献（advisor-panel 提到 Koustuv 可能会要）也还没读。
- 两轮合计 154 篇摘要级以上（第二轮新增 10 篇，另有 PersonaMem 和 Alexandria 两篇是重读；6 篇读了全文或全文相关章节）。下面只写对本项目有用的结论，不是综述。

## 1. 两位老师的工作里对本项目最要紧的

| 论文 | 与本项目的关系 |
|---|---|
| Pal, Goyal, Chandrasekharan, Saha 2026 *Algorithmic Cultivation*（arXiv 2605.17010） | **同一份数据**。treatment = 首次与 feed 发生任何互动（bookmark/点赞/评论/转发/引用/发帖）；PSM（AdaBoost，15 层）；结果 = 语言向 feed 靠拢。只比较 post 期的 ATE，没有 event-study 和 pre-trend 图。本项目必须避开"feed → 语言"，并且把"显式表态"和"隐式互动"拆开。RQ7 是对它的 event-study 补充，不是重复。 **第二轮（A8）在它的 treatment 定义下重做了逐周曲线和 placebo：方向与它一致，Science 上没有事前趋势；第一轮 RQ7"背书后没有趋同"的说法已撤回（REPORT §2.7）。** |
| Pal et al. 2026 *Hidden Toll of Social Media News*（EMNLP） | 同一数据的 News feed；发现"bookmark News feed"与心理语言恶化的关联比评论/引用大十倍以上。说明组里已经把 bookmark 当作一种 engagement 类型，但没有问它和实际互动是否一致。 |
| Yuan et al. 2026 *Mental Health Impacts of AI Companions*（CHI） | **本项目的模板**：Reddit 准实验（分层 PSM + DiD）+ 18 个半结构化访谈，三角验证后给设计建议。 |
| Lambert, Saha, Chandrasekharan 2025 *Does Positive Reinforcement Work?*（CHI） | PSM + DiD，目视检查 parallel trends，Bonferroni。我们的 event-study 是同一套路的周粒度版本。 |
| Saha et al. 2024 *Observer Effect*（CHI） | placebo 日期检验（150 个随机日期）；我们照做。 |
| Saha & Chandrasekharan 2026 *From Web(logs) to Web(AI)* | 报告清单：谁的标签定义了参照、placebo/negative control、缺失。REPORT 按它自查。 |
| Chan et al. 2024/2026（r/popular）；Chandrasekharan et al. 2017/2022（ban、quarantine） | 平台设计干预 → 用户行为的因果估计；DiD、ITS、对照社区 bootstrap。 |
| Goyal et al. 2026 *Hidden Values*；*VASTU* | 旧方案里 value 抽取的来源。VASTU 的结论（prompted LLM 识别社区价值只有 0.60 AUROC）与我们旧结果（Qwen/GPT 一致率 52%）一致：value 标注不适合做主测量。 |
| Koshy et al. 2023 *User-Moderator Alignment*（CMV） | "政策支持高、实践支持低"：说的和做的不一致，在治理语境下的先例。 |
| Goyal, Choi, Chandrasekharan *PILOT* | Eshwar 组在 Bluesky 上做 feed 编写系统；本项目给它提供"用户实际怎么用现有 feed"的观察证据。 |
| 其余（Saha 2018/2019/2020/2021/2025，Yuan 2023，Jhaver 2024，Choi 2025，Goyal 2026 Language of Approval 等） | 分层 PSM、SMD 阈值（0.15–0.3）、ITE 回归、BH-FDR、敏感性分析的具体做法。 |

## 2. 外部文献给出的定位

**(a) 没人看过订阅者一侧。** Zhou, Wang & Zhang 2026 访谈 26 位 feed 创建者并分析 88,302 个 feed，局限里明确写"没有直接征求 feed 消费者的看法"，
并且因为 AT Protocol 不暴露 pin/view，只能"用点赞 feed 的用户作为订阅者的代理"。Quelle & Bovet 2025 只报告"custom feed 的采用有限"。
Paper Skygest（Greenwood & Garg 2026）有服务器日志，报告首次使用后一周内论文类帖子占点赞比例上升 3.5 个百分点，但只有一个 feed、没有对照组、只看一周。
Bonsai、Feed Elicitation、PILOT、Braids 都是系统 + 小样本用户研究。
→ 空白：**多 feed、带对照、带 pre-trend、8 周跨度的"显式选择 vs 实际互动"观察研究，加消费者访谈。**

**(b) 理论：说的和做的。** Kleinberg, Mullainathan & Raghavan 2023：平台用 revealed preference 推断用户想要什么，这个假设本身有问题。
Milli et al. 2025：按 stated preference 排序会减少愤怒/敌意内容。Milkman, Rogers & Bazerman 2009："should" 类 DVD 被租来却迟迟不看。
Ekstrand & Willemsen 2016、Lyngs et al. 2018、Stray et al. 2023 都主张把显式表达纳入系统。
但这些要么是理论，要么是实验室/问卷里的 stated preference；**平台上自然发生的显式选择（点赞一个算法）和随后的行为之间的差距，没有被量过。**
Vaccaro et al. 2018 进一步提示：控制项本身就有安慰剂效应，用户"有了选择"就更满意，不管它起不起作用。

**(c) selection vs influence。** Aral et al. 2009（同质性解释了一半以上的"传染"）、Shalizi & Thomas 2011（观察数据里两者一般不可分）、
Lewis et al. 2012、Crandall et al. 2008、Slater 2007（reinforcing spirals：选择和效应互相强化）。
Chen et al. 2023 与 Hosseinmardi et al. 2021：YouTube 上极端内容的消费主要来自订阅和外链，即用户自己的选择，而不是推荐。
Bakshy et al. 2015、González-Bailón et al. 2023：个人选择比算法更限制跨立场接触。
→ 对"feed 塑造用户"的任何说法，都要先回答"他们是不是本来就在往那边走"。Roth 2022 提醒 pre-trend 检验功效低，所以我们同时报告只用早期历史匹配的规格。

**(d) feed 类型。** Ren, Kraut & Kiesler 2007 的 common identity vs common bond；Zhou et al. 2026 的 utility-providing vs community-building。
Blacksky（身份/社区型）与 Science、News（话题/工具型）应分开报告，不合并。

**(e) 方法。** Kiciman et al. 2018、De Choudhury & Kiciman 2017、Olteanu et al. 2017（社交媒体上的分层 PSM）；
Sun & Abraham、Callaway & Sant'Anna（交错处理时 TWFE 有偏，所以我们按日历周分 cohort、只用 never-treated 做对照）；
VanderWeele & Ding 2017（E-value）。


## 3. Personalization 与合成数据（第二批）给出的启发

- **BELA**（Benchmarking In-context Experiential Learning Through Repeated Product Recommendations, arXiv 2511.22130）：真实 Amazon 商品 + 合成 persona（已知潜在偏好）+ LLM 顾客模拟器；作者明说目的不是复现真实消费者，而是提供"可控、可扩展的测试台"。
- **Mem-PAL**（arXiv 2511.13410）：没有真实长期交互数据时，用多步 LLM 合成 + 人工校验得到 PAL-Set。
- **AlpsBench**（arXiv 2603.26680）反过来指出合成对话"过于显式"，与真实对话有分布差距；**Wang et al. 2025** 指出 LLM 扮演人群会抹平群体内差异；**Dong et al. 2024**（Can LLM be a Personalized Judge?）发现按 persona 让 LLM 当裁判并不可靠。
- 本项目的用法（Line C，见 REPORT §3）：只借 BELA 的"已知真值"这一点。合成用户的显式 profile 是我们种下的，行为由参数化选择模型在**真实 Science feed 帖子**上产生，
  不让 LLM 扮演用户（避免"LLM 推断 LLM 生成的行为"的循环）。这样才能回答旧 proposal 5.1/5.2 里真实数据答不了的两个问题：推断是不是瓶颈、保真度是否预测效用。
- **HyPerAlign、PRELUDE、PrefEval、PersonaMem、LaMP/LongLaMP、PersonalLLM**：都是从历史推断显式偏好再使用。Line B 的结果（推断出的 profile 不比原始历史多信息）对这一类方法在 feed 排序上的适用性是一个反例，值得在论文 related work 里点明。
- **Hou et al. 2023（LLM zero-shot ranker）、Kang et al. 2023、ONCE、Radlinski et al. 2022、Agent4Rec、Törnberg et al. 2023、Y Social**：LLM 排序/用户模拟的基线与测试台，供后续扩展。

## 4. 第二轮补读给出的启发（2026-11-14）

- **Epstein et al. 2025（Whose Values?）直接解释了旧项目里 value 标注为什么不稳。** 他们让 1,079 人给 5,211 条帖子标 Schwartz 价值（32,370 条标注），
  人与人之间一致性低，人与 LLM 之间也低；解决办法是让每个用户先做少量校准标注，再做**个性化**的价值分类器。
  旧 PRISM 里 Qwen 与 GPT 的精确一致率只有 52%、value profile 的排序 AUC 只有 0.54，与这个结论一致：帖子表达了什么价值本身是主观的，
  不存在一个可以从行为里推断出来、对所有人都成立的 value 标签。他们的修复同样依赖用户**自己给出**的显式输入。
- **Park et al. 2024**：用两小时访谈（自述）构建的 agent 在 GSS 上达到受试者自身两周重测一致性的 83–86%，只用人口学信息是 74%。
  和上一条、和 Line B/C 的结论是同一个方向：有信息量的是用户自己说的内容，不是从外部特征或行为替他们推断的摘要。
- **Jahanbakhsh et al. 2025、Alexandria**：让用户对价值设权重来重排 feed（N=141、250；N=12、257），评估的是 feed 是否符合用户声明的价值，
  没有评估用户之后是否真的按这些价值去互动。本项目的"说的和做的不重合"正好是这类 stated-preference 控件缺的那一半证据。
- **TEARS**：LLM 写的文本用户摘要要和协同过滤的 VAE 表征对齐之后才能持平或超过基线；卖点是可检查、可编辑，不是更准。和 Line B 的结果（推断出的 profile 不如原始历史）一致。
- **BELA（全文）**：真实 Amazon 商品（7.1 万件、2 千个选择集）+ 100 万个合成 persona + LLM 顾客模拟器；每个 persona 对每件商品的"真值"偏好由 GPT-4o 和 Gemini-1.5-Pro 打 0–100 分取平均，指标是 regret。
  作者明说"designed for learnability rather than realism"，并在局限里写不期望模拟顾客反映真实用户。可借的三点：
  (1) 打真值分的模型和当 oracle 的模型分开（oracle 用 Claude Sonnet 4），避免循环；Line C 现在模拟器和打分器用同一个 embedding 空间，正是这个问题；
  (2) 一致性检查（三个种子的打分波动 4.1，远小于类内标准差 25.4；模拟回答与 persona 的一致率、重复询问的稳定率 94.4%）；
  (3) 用人工挑选的"理想轨迹"证明环境里确实存在可学的跨 episode 信息，再说模型没学到。结论是前沿模型能在一个对话内学习，但不会跨 episode 积累。
- **Mem-PAL（全文）**：100 个合成用户，画像 → persona（时间线 + 需求类型与偏好）→ 每个 session 的情境 → 日志和对话，逐级由 Qwen2.5-Max 生成，再做规则校验和人工修订；
  按时间切分，最后一个月当评测。消融结果里，去掉具体的历史对话提纲（M_T）掉得最多（选择得分 38.3 → 28.1），去掉抽象的偏好原则（M_P）只掉到 36.3；
  session 级检索基线就有 33.8。也就是说，即使在按设定种下偏好的合成数据上，**检索具体历史也比抽象出的偏好原则有用**，和 Line B 的 implicit-knn > 推断 profile 同向。
- **Argyle 2022、Santurkar 2023、Social Simulacra、Xie 2024、PersonaChat、PersonaMem、MIND**：模拟人群与个性化基准的背景文献，对本项目主线没有新的约束。

## 5. 完整阅读清单（A–K）

### A. Bluesky 与去中心化平台

- Andrea Failla, Giulio Rossetti (2024). “I’m in the Bluesky Tonight”: Insights from a year worth of social data. *PLoS ONE*.
- Leonhard Balduf, Saidu Sokoto et al. (2024). Looking AT the Blue Skies of Bluesky. *ACM Internet Measurement Conference (IMC)*.
- Martin Kleppmann, Paul Frazee et al. (2024). Bluesky and the AT Protocol: Usable Decentralized Social Media. *preprint*.
- Dorian Quelle, Alexandre Bovet (2025). Bluesky: Network topology, polarization, and algorithmic curation. *PLoS ONE*.
- Carlo Alberto Bono, Chang Liu et al. (2026). Self-moderation in the Decentralized Era: Decoding Blocking Behavior on Bluesky. *Proceedings of the International AAAI Conference on Web and Social Media*.
- Yuhan Liu, Emmy Song et al. (2025). Understanding Decentralized Social Feed Curation on Mastodon. *Proceedings of the ACM on Human-Computer Interaction*.
- Ujun Jeong, Paras Sheth et al. (2024). Exploring Platform Migration Patterns between Twitter and Mastodon: A User Behavior Study. *Proceedings of the International AAAI Conference on Web and Social Media*.
- Lucio La Cava, Luca Maria Aiello et al. (2023). Drivers of social influence in the Twitter migration to Mastodon. *Scientific Reports*.
- Jiahui He, Haris Bin Zia et al. (2023). Flocking to Mastodon: Tracking the Great Twitter Migration. *ACM Internet Measurement Conference (IMC)*.
- Casey L. Fiesler, Brianna Dym (2020). Moving Across Lands: Online Platform Migration in Fandom Communities. *Proceedings of the ACM on Human-Computer Interaction*.

### B. 用户对算法的控制、custom feed、middleware

- K. J. Kevin Feng, Xander Koo et al. (2024). Mapping the Design Space of Teachable Social Media Feed Experiences. *CHI Conference on Human Factors in Computing Systems*.
- Omar El Malki, Marianne Aubin Le Quéré et al. (2026). Bonsai: Intentional and Personalized Social Media Feeds. *CHI Conference on Human Factors in Computing Systems*.
- Jenna Burrell, Zoe Kahn et al. (2019). When Users Control the Algorithms: Values Expressed in Practices on Twitter. *Proceedings of the ACM on Human-Computer Interaction*.
- F. Maxwell Harper, Funing Xu et al. (2015). Putting Users in Control of their Recommendations. *ACM Conference on Recommender Systems (RecSys)*.
- Kai Lukoff, Ulrik Lyngs et al. (2021). How the Design of YouTube Influences User Sense of Agency. *CHI Conference on Human Factors in Computing Systems*.
- Shagun Jhaver, Alice Qian Zhang et al. (2023). Personalizing Content Moderation on Social Media: User Perspectives on Moderation Choices, Interface Design, and Labor. *Proceedings of the ACM on Human-Computer Interaction*.
- Rahul Bhargava, A. J. Chung et al. (2019). Gobo: A System for Exploring User Control of Invisible Algorithms in Social Media. *ACM Conference on Computer Supported Cooperative Work (CSCW)*.
- Kilian Feiertag (2026). Personalizing Content Moderation: How Middleware Shapes User Perception and Experience on Social Media. *reposiTUm (TU Wien)*.
- Kristen Vaccaro, Dylan Huang et al. (2018). The Illusion of Control: Placebo Effects of Control Settings. *CHI Conference on Human Factors in Computing Systems*.
- Amy Xian Zhang, Michael S. Bernstein et al. (2024). Form-From: A Design Space of Social Media Systems. *Proceedings of the ACM on Human-Computer Interaction*.
- Shagun Jhaver, Seth Frey et al. (2023). Decentralizing Platform Power: A Design Space of Multi-Level Governance in Online Social Platforms. *Social Media + Society*.

### C. stated vs revealed preference、价值对齐的排序

- Jon Michael Kleinberg, Sendhil Mullainathan et al. (2023). The Challenge of Understanding What Users Want: Inconsistent Preferences and Engagement Optimization. *Management Science*.
- Smitha Milli, Micah Carroll et al. (2025). Engagement, user satisfaction, and the amplification of divisive content on social media. *PNAS Nexus*.
- Jonathan Stray, Alon Y. Halevy et al. (2023). Building Human Values into Recommender Systems: An Interdisciplinary Synthesis. *ACM Transactions on Recommender Systems*.
- Chenyan Jia, Michelle S. Lam et al. (2024). Embedding Democratic Values into Social Media AIs via Societal Objective Functions. *Proceedings of the ACM on Human-Computer Interaction*.
- Akaash Kolluri, Renn Su et al. (2026). Alexandria: A Library of Pluralistic Values for Realtime Re-Ranking of Social Media Feeds. *Proceedings of the International AAAI Conference on Web and Social Media*.
- Tiziano Piccardi, Martin Saveski et al. (2025). Reranking partisan animosity in algorithmic social media feeds alters affective polarization. *Science*.
- Michael D. Ekstrand, Martijn C. Willemsen (2016). Behaviorism is Not Enough: Better Recommendations through Listening to Users. *ACM Conference on Recommender Systems (RecSys)*.
- Ulrik Lyngs, Reuben Binns et al. (2018). "So, Tell Me What Users Want, What They Really, Really Want!". *CHI Conference Extended Abstracts on Human Factors in Computing Systems (CHI EA)*.
- Smitha Milli, Luca Belli et al. (2021). From Optimizing Engagement to Measuring Value. *ACM Conference on Fairness, Accountability, and Transparency (FAccT)*.
- Tom Cunningham, Sana Pandey et al. (2025). Ranking by engagement and non‐engagement signals: Learnings from industry. *Annals of the New York Academy of Sciences*.
- Katherine L. Milkman, Todd Rogers et al. (2008). Harnessing Our Inner Angels and Demons: What We Have Learned About Want/Should Conflicts and How That Knowledge Can Help Us Reduce Short-Sighted Decision Making. *Perspectives on Psychological Science*.
- Katherine L. Milkman, Todd Rogers et al. (2009). Highbrow Films Gather Dust: Time-Inconsistent Preferences and Online DVD Rentals. *Management Science*.
- Ashton Anderson, Lucas Maystre et al. (2020). Algorithmic Effects on the Diversity of Consumption on Spotify. *International World Wide Web Conference (WWW)*.
- Tan Zhi‐Xuan, Micah Carroll et al. (2024). Beyond Preferences in AI Alignment. *Philosophical Studies*.
- Hunt Allcott, Luca Braghieri et al. (2020). The Welfare Effects of Social Media. *American Economic Review*.
- Hunt Allcott (2020). Digital Addiction. *AEA Randomized Controlled Trials*.
- Dimitris Kalimeris, Smriti Bhagat et al. (2021). Preference Amplification in Recommender Systems. *ACM SIGKDD Conference on Knowledge Discovery and Data Mining (KDD)*.
- Micah Carroll, Anca D. Dragan et al. (2022). Estimating and Penalizing Induced Preference Shifts in Recommender Systems. *arXiv (Cornell University)*.
- Allison J. B. Chaney, Brandon Stewart et al. (2018). How algorithmic confounding in recommendation systems increases homogeneity and decreases utility. *ACM Conference on Recommender Systems (RecSys)*.

### D. 算法的用户理解（folk theories）

- Motahhare Eslami, Aimee Rickman et al. (2015). "I always assumed that I wasn't really that close to [her]": Reasoning about Invisible Algorithms in News Feeds. *CHI Conference on Human Factors in Computing Systems*.
- Michael Ann DeVito, Darren Gergle et al. (2017). "Algorithms ruin everything": #RIPTwitter, Folk Theories, and Resistance to Algorithmic Change in Social Media. *CHI Conference on Human Factors in Computing Systems*.
- Nadia Karizat, Daniel Delmonaco et al. (2021). Algorithmic Folk Theories and Identity: How TikTok Users Co-Produce Knowledge of Identity and Engage in Algorithmic Resistance. *Proceedings of the ACM on Human-Computer Interaction*.
- Ellen Simpson, Bryan Semaan (2021). For You, or For"You"? Everyday LGBTQ+ Encounters with TikTok. *Proceedings of the ACM on Human-Computer Interaction*.
- Motahhare Eslami, Karrie G. Karahalios et al. (2016). First I "like" it, then I hide it: Folk Theories of Social Feeds. *CHI Conference on Human Factors in Computing Systems*.
- Emilee J. Rader, Rebecca Gray (2015). Understanding User Beliefs About Algorithmic Curation in the Facebook News Feed. *CHI Conference on Human Factors in Computing Systems*.
- Taina Bucher (2016). The algorithmic imaginary: exploring the ordinary affects of Facebook algorithms. *Information Communication & Society*.
- Kjerstin Thorson, Chris Wells (2015). Curated Flows: A Framework for Mapping Media Exposure in the Digital Age. *Communication Theory*.
- Joëlle Swart (2021). Experiencing Algorithms: How Young People Understand, Feel About, and Engage With Algorithmic News Selection on Social Media. *Social Media + Society*.

### E. selection vs influence、选择性接触

- Sinan Aral, Lev Muchnik et al. (2009). Distinguishing influence-based contagion from homophily-driven diffusion in dynamic networks. *Proceedings of the National Academy of Sciences*.
- Cosma Rohilla Shalizi, Andrew C. Thomas (2011). Homophily and Contagion Are Generically Confounded in Observational Social Network Studies. *Sociological Methods & Research*.
- Cristian Danescu-Niculescu-Mizil, Robert West et al. (2013). No country for old members: user lifecycle and linguistic change in online communities. *International World Wide Web Conference (WWW)*.
- Kevin Lewis, Marco Gonzalez et al. (2011). Social selection and peer influence in an online social network. *Proceedings of the National Academy of Sciences*.
- David Crandall, Dan Cosley et al. (2008). Feedback effects between similarity and social influence in online communities. *ACM SIGKDD Conference on Knowledge Discovery and Data Mining (KDD)*.
- Annie Y. Chen, Brendan J Nyhan et al. (2023). Subscriptions and external links help drive resentful users to alternative and extremist YouTube channels. *Science Advances*.
- Daniel Muise, Homa Hosseinmardi et al. (2022). Quantifying partisan news diets in Web and TV audiences. *Science Advances*.
- Eytan Bakshy, Solomon Messing et al. (2015). Exposure to ideologically diverse news and opinion on Facebook. *Science*.
- Homa Hosseinmardi, Amir Ghasemian et al. (2021). Examining the consumption of radical content on YouTube. *Proceedings of the National Academy of Sciences*.
- Brendan J Nyhan, Jaime E. Settle et al. (2023). Like-minded sources on Facebook are prevalent but not polarizing. *Nature*.
- Sandra González‐Bailón, David M. J. Lazer et al. (2023). Asymmetric ideological segregation in exposure to political news on Facebook. *Science*.
- Michael D. Slater (2007). Reinforcing Spirals: The Mutual Influence of Media Selectivity and Media Effects and Their Impact on Individual Behavior and Social Identity. *Communication Theory*.
- Manoel Horta Ribeiro, Veniamin Veselovsky et al. (2023). The Amplification Paradox in Recommender Systems. *Proceedings of the International AAAI Conference on Web and Social Media*.
- Manoel Horta Ribeiro, Raphael Ottoni et al. (2020). Auditing radicalization pathways on YouTube. *ACM Conference on Fairness, Accountability, and Transparency (FAccT)*.
- Jonah A. Berger, Chip Heath (2007). Where Consumers Diverge from Others: Identity Signaling and Product Domains. *Journal of Consumer Research*.
- Judith E. Möller (2021). Filter bubbles and digital echo chambers 1. *Routledge eBooks*.
- Isaac Waller, Ashton Anderson (2020). Community embeddings reveal large-scale cultural organization of online platforms. *arXiv (Cornell University)*.

### F. 准实验方法（PSM / DiD / event study / 敏感性）

- Emre Kıcıman, Scott Counts et al. (2018). Using Longitudinal Social Media Analysis to Understand the Effects of Early College Alcohol Use. *Proceedings of the International AAAI Conference on Web and Social Media*.
- Munmun De Choudhury, Emre Kıcıman (2017). The Language of Social Support in Social Media and Its Effect on Suicidal Ideation Risk. *Proceedings of the International AAAI Conference on Web and Social Media*.
- Alexandra Olteanu, Onur Varol et al. (2017). Distilling the Outcomes of Personal Experiences: A Propensity-scored Analysis of Social Media. *ACM Conference on Computer Supported Cooperative Work (CSCW)*.
- Jonathan Roth (2022). Pretest with Caution: Event-Study Estimates after Testing for Parallel Trends. *American Economic Review Insights*.
- Liyang Sun, Sarah Abraham (2018). Estimating Dynamic Treatment Effects in Event Studies with Heterogeneous Treatment Effects. *RePEc: Research Papers in Economics*.
- Tyler J. VanderWeele, Peng Ding (2017). Sensitivity Analysis in Observational Research: Introducing the E-Value. *Annals of Internal Medicine*.
- Manoel Horta Ribeiro, Homa Hosseinmardi et al. (2023). Deplatforming did not decrease Parler users’ activity on fringe social media. *PNAS Nexus*.
- Manoel Horta Ribeiro, Shagun Jhaver et al. (2021). Do Platform Migrations Compromise Content Moderation? Evidence from r/The_Donald and r/Incels. *Proceedings of the ACM on Human-Computer Interaction*.
- Giuseppe Russo, Luca Verginer et al. (2023). Spillover of Antisocial Behavior from Fringe Platforms: The Unintended Consequences of Community Banning. *Proceedings of the International AAAI Conference on Web and Social Media*.

### G. 社区认同、参与、留存

- Justine Zhang, William L. Hamilton et al. (2017). Community Identity and User Engagement in a Multi-Community Landscape. *Proceedings of the International AAAI Conference on Web and Social Media*.
- Chenhao Tan, Lillian Lee (2015). All Who Wander: On the Prevalence and Characteristics of Multi-community Engagement. *International World Wide Web Conference (WWW)*.
- William L. Hamilton, Justine Zhang et al. (2017). Loyalty in Online Communities. *Proceedings of the International AAAI Conference on Web and Social Media*.
- Yuqing Ren, Robert E. Kraut et al. (2007). Applying Common Identity and Bond Theory to Design of Online Communities. *Organization Studies*.
- Jennifer Preece, Ben Shneiderman (2009). The Reader-to-Leader Framework: Motivating Technology-Mediated Social Participation. *AIS Transactions on Human-Computer Interaction*.
- Eva Sharma, Munmun De Choudhury (2018). Mental Health Support and its Relationship to Linguistic Accommodation in Online Communities. *CHI Conference on Human Factors in Computing Systems*.
- Yiqing Hua, Mor Naaman et al. (2020). Characterizing Twitter Users Who Engage in Adversarial Interactions against Political Candidates. *CHI Conference on Human Factors in Computing Systems*.
- Andrew M. Guess, Neil A. Malhotra et al. (2023). How do social media feed algorithms affect attitudes and behavior in an election campaign?. *Science*.
- Ferenc Huszár, Sofia Ira Ktena et al. (2021). Algorithmic amplification of politics on Twitter. *Proceedings of the National Academy of Sciences*.
- Jerome Ramos, Hossein A. Rahmani et al. (2024). Transparent and Scrutable Recommendations Using Natural Language User Profiles. *Proceedings of the conference - Association for Computational Linguistics. Meeting*.
### H. arXiv 预印本（摘要）
- Popowski et al. (2026). Social Media Feed Elicitation. arXiv 2602.18594.
- Greenwood & Garg (2026). Paper Skygest: Personalized Academic Recommendations on Bluesky. arXiv 2601.04253.（读了采纳分析与局限）
- Zhou, Wang & Zhang (2026). Middleware for Feed Recommendation in Practice. arXiv 2609.12958.（读了方法与局限）
- Goyal, Choi & Chandrasekharan. PILOT: Control Surfaces for Authoring Social Media Feeds.

### I. 两位老师的论文（本地语料库 pid）
pal2026algorithmic, pal2026hidden, yuan2026mental, chan2026examining, goyal2026uncovering, goyal2026vastu, saha2024observer,
lambert2025positive, pal2026x, saha2026from, chan2024understanding, saha2025mentalb, yuan2023mental, jhaver2024bystanders,
ligo2025user, choi2025creator, goyal2026language, koshy2023measuring, chandrasekharan2022quarantined, li2025emails,
chandrasekharan2017you, saha2019social, saha2020causal, saha2018social, saha2021advertiming, lambert2024investigating,
chandrasekharan2018internet, saha2020psychosocial, goyal2026social；另有 zhang2026ai 只有题目（语料库无摘要），不计入。

### J. 第二批清单（摘要）
- LaMP: When Large Language Models Meet Personalization
- LongLaMP: A Benchmark for Personalized Long-form Text Generation
- PersonalLLM: Tailoring LLMs to Individual Preferences
- UserToolBench: A User-Profile-Hidden Benchmark for Personalized Decision Making in Tool-Use LLMs
- Do LLMs Recognize Your Preferences? Evaluating Personalized Preference Following in LLMs
- Aligning LLM Agents by Learning Latent Preference from User Edits
- HyPerAlign: Interpretable Personalized LLM Alignment via Hypothesis Generation
- Pairwise Calibrated Rewards for Pluralistic Alignment
- Can LLM be a Personalized Judge?
- Large Language Models as Zero-Shot Conversational Recommenders
- Large Language Models are Zero-Shot Rankers for Recommender Systems
- ONCE: Boosting Content-based Recommendation with Both Open- and Closed-source Large Language Models
- On Natural Language User Profiles for Transparent and Scrutable Recommendation
- Do LLMs Understand User Preferences? Evaluating LLMs On User Rating Prediction
- On Generative Agents in Recommendation
- Large language models that replace human participants can harmfully misportray and flatten identity groups
- A Roadmap to Pluralistic Alignment
- Personalization of Large Language Models: A Survey
- Simulating Social Media Using Large Language Models to Evaluate Alternative News Feed Algorithms
- PERSONA: A Reproducible Testbed for Pluralistic Alignment
- Y Social: an LLM-powered Social Media Digital Twin
- Mem-PAL (arXiv 2511.13410)；PersonaMem (2504.14225)；AlpsBench (2603.26680)；IBA-Bench (2608.02171)；BELA (2511.22130)

第一轮未读的各项，第二轮的处理见清单 K。

### K. 第二轮补读（2026-11-14）
摘要：
- Park, Zou, Kamphorst et al. (2024). LLM Agents Grounded in Self-Reports Enable General-Purpose Simulation of Individuals（即原题 Generative Agent Simulations of 1,000 People，arXiv 2411.10109 v3 改了标题）。
- Argyle, Busby, Fulda et al. (2022). Out of One, Many: Using Language Models to Simulate Human Samples. arXiv 2209.06899.
- Santurkar, Durmus, Ladhak et al. (2023). Whose Opinions Do Language Models Reflect? arXiv 2303.17548.
- Park, Popowski, Cai et al. (2022). Social Simulacra. arXiv 2208.04024.
- Xie, Chen, Jia et al. (2024). Can Large Language Model Agents Simulate Human Trust Behavior? arXiv 2402.04559.
- Zhang, Dinan, Urbanek et al. (2018). Personalizing Dialogue Agents: I have a dog, do you have pets too? arXiv 1801.07243.
- Penaloza, Gouvert, Wu, Charlin (2024). TEARS: Textual Representations for Scrutable Recommendations. arXiv 2410.19302.
- Jiang, Hao, Cho et al. (2025). Know Me, Respond to Me (PersonaMem). arXiv 2504.14225.（第一轮按 arXiv 号列过，这次读的是完整摘要，不重复计数）
- Kolluri, Su, Jahanbakhsh et al. (2025). Alexandria. arXiv 2505.10839.（第一轮读过正式版摘要，不重复计数）
- Epstein, Jahanbakhsh, Piccardi et al. (2025). Whose Values? Measuring the (Subjective) Expression of Basic Human Values in Social Media. arXiv 2511.08453.
- Jahanbakhsh, Zhao, Piccardi et al. (2025). Value Alignment of Social Media Ranking Algorithms. arXiv 2509.14434.
- Wu, F. et al. (2020). MIND: A Large-scale Dataset for News Recommendation. ACL 2020.（摘要取自 ACL Anthology）
全文（正文，不含附录）：
- BELA: Benchmarking In-context Experiential Learning Through Repeated Product Recommendations. arXiv 2511.22130.
- Mem-PAL: Towards Memory-based Personalized Dialogue Assistants for Long-term User-Agent Interaction. arXiv 2511.13410.
没读到：STELA；Enabling User Agency in Scalable Content Recommendations；Li et al. AAMAS 2026。
