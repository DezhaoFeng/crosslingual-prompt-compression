# 项目执行计划与测试节点（Gates）

以 `proposal/proposal final/main.tex` 为准（取代 `finetune_plan.md` 中已过时的部分）。

- **Core**（2026-11-03 缩减，见 `proj/DECISIONS.md`“范围决定”）：M0、M1（zh）、M4（en control）× ρ ∈ {0.5, 0.33, 0.2} × Belebele en/zh，检验限于中文的 H1、H2
- **Stretch**（Core 完成后再做）：sw 全量标注和 M2（sw）、M3（zh+sw）、sw 上的探索性迁移分析、H3 迁移到 Yoruba/Lithuanian、第二个 reader、abstractive rewriting
- 下面各 Gate 中涉及 sw 和 M2 的要求，都按这个范围理解：sw 不阻塞 Core
- **全量生成的前提**：en/zh 通过未参与调参的新样本验证

## 原则

1. **先小后大**：几十个样本先把整条链路跑通（teacher → labels → fine-tune → compress → reader → metric），再做全量
2. **Gate 检查的是工程和数据质量，不是假设成立**：gate 不通过就不进下一步；H1/H2 不成立是有效的负结果，照常报告
3. **Belebele 只当测试集**：
   - 压缩算法（含中文删除单位）、prompt、训练参数的调整，以及 checkpoint 选择，一律用 Wikipedia dev split
   - Belebele 只允许两种 pilot 用途：G0 的压缩率诊断（不看答案）和 G1 预先规定的 reader 资格检查。不能用它比较哪个 reader 对 H1/H2 更有利
4. **公平比较**：主比较中的 M0、M1、M2、M4 使用同一套 inference（分词、分数聚合、删除规则）、同一个 reader、同一组 passage
5. **Eval harness 先冻结再看结果**：M1 训练完之前就固定好；之后修 bug 必须记录版本，并重跑受影响的结果

## 时间线

| 时间 | 阶段 | 包含的 Gate |
|---|---|---|
| 10/30–11/4 | End-to-end pilot；11/4 做 zh/sw go/no-go | G0–G4 |
| 11/5–11/11 | 全量 label distillation；训练 M1、M2、M4 | G5、G6、G7 |
| 11/12–11/25 | Core evaluation + error analysis；**milestone** | Milestone 交付 |
| 11/25 之后 | Stretch 项目；poster | — |

**2026-11-03 更新：11/4 的 go/no-go 结论是 no-go**（`proj/DECISIONS.md`“en/zh 新样本验证结果”）。11/5 的全量蒸馏及后续 M1、M4 训练日期需要重新安排；新的开始日期取决于替代方案通过验证。

## 定义

- **ρ（requested rate）**：保留**词数**的比例，即保留 ⌈ρn⌉ 个词（proposal 中的定义）
- **Word retention**：实际保留词数 / 原词数。用来检查是否遵守预算
- **Reader-token retention**：压缩后与原文的 reader tokenizer token 数之比。只作报告和异常检查，**不要求接近 ρ**
- 以下所有 ±x 都指**百分点**

## Gates

以下阈值是建议默认值，proposal 里没有规定；pilot 之后可以调整，但调整要记录在本文件。

### G0 环境 + M0 实际压缩率（10/31，负责：Jonathan）
- 检查：M0 能在 4090 上加载；Belebele en/zh/sw 每种语言 900 题、488 个 passage
- 检查：用原版 M0 + 官方 `rate` 压缩全部 passage（只看压缩率，不看答案），按语言和 ρ 记录 word retention 和 reader-token retention 的分布，标出异常（例如中文整句被当成一个"词"，导致全留或全删）。这是诊断，不设阈值
- 检查：用我们自己的推理（`proj/DECISIONS.md` B2），word retention 必须恰好等于 ⌈ρn⌉/n
- 中文没通过时：**在 Wikipedia zh dev 数据上**修复删除单位（Risk 1）
  - 保留原版 M0 的结果作为**诊断结果**，写进报告
  - 修复后的 inference 用于 M0、M1、M2、M4 的**所有**主比较，否则分不清收益来自监督数据还是推理修复
- 这是最大的 go/no-go 风险

### G1 Reader capability check（11/1，负责：Jonathan）
- 资格标准事先固定：每种语言取同样的约 100 题，full − none ≥ 15 个百分点
- 按事先排好的顺序试候选 reader，**选第一个在三种语言上都通过的**；不能根据压缩模型的结果挑 reader
- 同时测吞吐和 API 成本，外推全量评测的花费
- 没有 reader 通过 → Swahili 不纳入 Core，并在报告中说明

### G2 Aligner 单元测试（11/1–11/2，负责：Dezhao）
用一个小的 `assert` 脚本或 `test_*.py` 覆盖：
- **Round-trip**：label=1 的词按原顺序拼回来，必须等于 teacher 的压缩输出（只删词的情况）
- 中文（jieba）、Swahili（按空白和标点切）、英语（原 spaCy 逻辑）各有几个手写例子
- **中文打标签时的切词单位必须和 G0 确定的 inference 删除单位一致**
- 通过：全部 assert 通过

### G3 Contamination filter 测试（11/2，负责：Dezhao）
- 先固定 normalization（大小写、标点、空白）、n-gram 大小和 overlap 阈值，写进代码和本文件
  - 默认值（待组员确认，详见 `proj/DECISIONS.md` B10）：NFKC + casefold + 去掉标点、符号、空白和控制字符；字符 n-gram，en/sw 50、zh 20；有 1 个相同 n-gram 就删除
- 测试用例：
  - 完整的 Belebele passage → 必须删除
  - 部分重叠（例如 passage 的一半拼进其他文本）→ 必须删除
  - 只改了标点和空白的 passage → 必须删除
  - 无关的 Wikipedia 文本 → 必须保留（检查误删）
- 记录：每种语言被删掉的 chunk 数和比例；抽看几条被删的 chunk，确认不是大面积误删

### G4 End-to-end pilot（11/3–11/4，全员）
每种语言几十个 chunk，完整跑一遍：
- Teacher 压缩 → 对齐 → 过滤（规则见下方"数据处理"）；记录每种语言的 variation rate、alignment gap，以及过滤前后的数量
- 训练 1 个 epoch：loss 下降；**中途 kill 掉进程，确认能从 checkpoint resume**（CSIL 凌晨 3:00 会重启）
- 压缩后的文本交给 reader，能得出数字
- 通过：整条链路跑通；zh/sw 的 variation rate 与 en 不是数量级的差距
- **11/4 zh/sw go/no-go**：teacher 偷偷改写 → 换 teacher 或者加 few-shot 示例

### G5 全量蒸馏之前（11/5，负责：Dezhao）
- **中文 100 条人工 audit 完成后**，才开始全量蒸馏
  - 单位：随机抽 100 个 chunk（native speaker 审核）
  - 错误类型：① teacher 违规（改写、换序、加词）；② 对齐错误（teacher 输出正确，但 label 错了）；③ 分词错误
  - 错误率 = 至少有一个 ②/③ 类错误的 chunk 数 / 100；① 类单独统计
  - 通过：错误率 < 10%（建议值）
- 用 pilot 实际消耗的 token 数确认 teacher 成本

### G6 Eval harness 冻结（M1 训练完之前，负责：Jonathan）
- U 的计算：分母接近 0 时给出警告或返回 NaN，不能输出一个巨大的数
- Random baseline 按**每个 passage 相同的词预算**做 budget-matched；随机种子固定并记录
- Bootstrap：
  - 按 **passage** 重采样（同一 passage 的题共用一份压缩结果）
  - 所有模型用**同一组重采样**（paired）
  - 直接计算**差值**的 95% CI，不只看各模型各自的 CI：
    - H1：acc(Mk) − acc(M0)，在目标语言上
    - H2：[acc(Mk) − acc(M0)] − [acc(M4) − acc(M0)]，在目标语言上
- **主要分析**（事先指定）：raw accuracy，ρ = 0.33（与 audit 的 one-third 对应）。其他 ρ 和 U 作为次要结果报告，并在报告中注明没有做多重比较校正
- Reader 用 temperature 0，所有输出缓存到磁盘
- 通过：用 M0 跑一遍，能复现 G0/G1 的数字
- 冻结之后修 bug：记录版本，重跑受影响的结果

### G7 每个模型训练完（11/5–11/11，负责：Jiehuan）
- Checkpoint 只按对应训练语言的 Wikipedia dev split 选
- M4 的训练预算（样本数、步数）与 M1/M2 一致
- G0 的压缩率检查对新模型再跑一遍
- 英语退化**只记录、不拦截**：在 Wikipedia en dev 上和 Belebele en 上与 M0 对比，结果写进报告分析

## 数据处理（proposal 规定）

- 规模：每种语言（zh、sw、en）约 2–3k 个 chunk，每个约 512 tokens，来自 `wikimedia/wikipedia`
- Train/dev 切分：**按 Wikipedia 原始文档**切，同一文档的 chunk 不跨 train/dev
- 污染过滤：见 G3，在切分之前做
- 质量过滤：先排除 aligner 判为非 `ok` 的样本，并按原因统计；`ok` 样本的指标全为 0 时全部保留，不按 10% 百分位删除。这一点**偏离 proposal**，原因见 `proj/DECISIONS.md` C2，报告中要说明
- zh：切 chunk 之前先用 OpenCC `t2s` 转成简体（`proj/DECISIONS.md` B5）
- 记录每一步的数量：原始 → 污染过滤后 → 质量过滤后 → train / dev

## Milestone 交付（11/25，全员）

这不是 gate：H1/H2 不成立也一样交付。

- 结果表：{M0 原版（诊断）, M0, M1, M2, M4, random, full, none, teacher} × ρ × 语言；给出 raw accuracy、U、word retention、reader-token retention、95% CI
- H1/H2：报告 G6 中定义的差值和 CI，判断是否获得支持；同时报告 English–target 差距的变化，避免把"英语变差"误读为"差距缩小"
- Error analysis：被删掉的是哪一类词

## 分工

| 成员 | 负责 |
|---|---|
| Dezhao Feng | Data pipeline：G2、G3、G4 的数据部分、G5、数据处理 |
| Jiehuan Zhao | Training：G4 的训练和 resume 测试、G7 |
| Jonathan Yang | Evaluation：G0、G1、G6（不依赖其他人，现在就能开始） |

## 待决定

- [ ] Teacher 用哪个模型（影响 G4 和成本）
- [ ] 候选 reader 及其尝试顺序（G1 之前固定）
- [ ] 确认主要分析选 raw accuracy @ ρ = 0.33
- [ ] 确认上面各项阈值，以及 G3 的 n-gram 大小和 overlap 阈值
- [ ] 确认 CSIL 工作站可以用于 713 项目（问助教）
