# 训练标签交接说明（给 Jiehuan）

2026-11-16，Dezhao。数据部分（G2–G5、全量 distillation）已完成，从这里开始是训练（G4 训练部分、G7）。
背景和所有决定都在 `proj/DECISIONS.md`，主要是“全量数据池与 teacher 批次规则”和“全量 distillation 结果”两节。

## 文件

| 文件 | 条数 | 用途 |
|---|---|---|
| `zh_train.jsonl` | 1567 | M1 的训练数据 |
| `en_train.jsonl` | 1567 | M4 的训练数据（数量和 zh 相同） |
| `zh_dev.jsonl` / `en_dev.jsonl` | 338 / 338 | 选 checkpoint，只按对应语言的 dev 选（plan.md G7） |
| `summary.json` | — | 每组的生成数、可用数、状态分布、保留率、标点占比、token 数 |

所有样本都来自 Wikipedia（`data/full/`），已经做过 Belebele 污染过滤，train 和 dev 按文档切分，不会有同一篇文档同时出现在两边。Belebele 只能当测试集，不能用来调参或选 checkpoint（plan.md 原则 3）。

## 每行的字段

- `text`：原文（zh 已转简体）
- `input_ids`：XLM-R token，**不含** `<s>` `</s>`，由 `segment.segment()` 生成，和推理时的切法完全相同
- `word_ids`：每个 token 属于第几个词
- `labels`：每个**词**一个标签，1 = 保留，0 = 删除
- `word_spans`：每个词在 `text` 中的字符位置；空 span 表示只有空白的词
- `prep_text` / `inserted_spaces`：zh 先用 jieba 插入空格再交给 XLM-R，这两个字段记录插入的位置
- `status`、`stats`、`teacher_output`：teacher 的原始输出和对齐状态，只用来追溯，训练用不到

词标签转成 token 标签：`segment.expand_labels(word_ids, labels)`。`<s>`、`</s>` 和 padding 由你在训练代码里加。它们的标签怎么处理（沿用官方训练脚本，还是设成 -100 忽略）还没定，在 DECISIONS.md C 节的待决定里，请你决定后写进去。

## 需要你做的

1. **链路验证（先做）**：用一小部分 train（例如 200 条）训练几十步，确认 loss 下降；中途 kill 掉进程，确认能从 checkpoint resume（CSIL 每天凌晨 3:00 重启）。这一步只确认流程能跑通，不要求效果
2. **M1 和 M4**：都从 M0（`microsoft/llmlingua-2-xlm-roberta-large-meetingbank`，revision 见 `segment.py`）继续训练，M1 用 zh_train，M4 用 en_train。两边的**样本数、步数、batch、学习率和序列长度都要相同**
3. **记录**：epoch、batch、步数、有效监督 token 数（不含特殊 token 和 padding）、每次 dev 评估的结果，以及最后选中的 checkpoint
4. **G7**：新模型训练完，用 G0 的压缩率检查再跑一遍；英语退化只记录、不拦截

## 已知偏差（报告里要写）

- **zh 可用率只有 58%**（2700 条生成，1567 条可用），en 是 85%。被过滤掉的 zh 样本压得更狠，所以留下来的 zh 标签**偏保守**：平均保留率 zh 0.72、en 0.68
- **zh 被删的词里约 28% 是标点**，en 只有约 3%。两种语言的监督内容不完全对称
- 中文审核（G5，zh train 第 31–130 条）：AI 初审了全部 59 条可用样本，母语者复核了其中 10 条片段，**没有发现标签错误**。确认了 2 条语义错误，都是 teacher 删掉了限定词（例如“抗日战争爆发后”变成“抗日战争后”）。按现有规则，这 2 条（`zh_3477_c0`、`zh_1311_c0`）仍然可用，**在 `zh_train.jsonl` 里**
- “可用”包括 `ok_relaxed`：teacher 输出和原文之间允许少量引号、空格、大小写差异（DECISIONS.md B3a）。标签总是从**原文**重建，所以训练文本就是原文，但 teacher 输出本身不一定逐字是原文的子序列
- 有 3 次 API 网络失败后重试成功，其中 1 次计费不明，标签不受影响（见 DECISIONS.md）
- teacher 用的是 DeepSeek API，API 背后的模型权重没办法固定版本，所以这些标签不能保证逐字复现。原始输出都保存在 `proj/results/teacher/full/`

## 复现

```bash
python sample_wiki.py --out <dir> --langs zh --train 3500 --dev 800 --seed 713   # en: --train 2500 --dev 600
python contamination.py --in <dir> --out <clean_dir> --langs zh
python distill_full.py --lang zh --split train                                    # 需要 DEEPSEEK_API_KEY
python export_labels.py
```
