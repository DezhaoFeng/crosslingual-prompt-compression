# 实验接口与决定（DECISIONS）

三个人共用的约定。修改任何一条都要在"变更记录"里写明日期、原因和受影响的结果。总体计划见 `plan.md`。

## A. 已固定

### 模型与代码版本
| 项目 | 固定为 |
|---|---|
| M0 | `microsoft/llmlingua-2-xlm-roberta-large-meetingbank`，revision `ebaba9b0e874dadd3003ffcff828e4397e568089` |
| LLMLingua 代码 | `microsoft/LLMLingua` commit `5a4c78ae18ab17a98cf997e8259354e546081d64`（2026-09-10） |
| Wikipedia | `wikimedia/wikipedia`，配置 `20231101.en / .zh / .sw`，revision `b04c8d1ceb2f5cd4588862100d08de323dccfbaa` |

- **代码只用一个来源**：`pip install git+https://github.com/microsoft/LLMLingua@5a4c78ae18ab17a98cf997e8259354e546081d64`，不用 PyPI 上的 `llmlingua==0.2.2`
- 复用的官方函数（如 `is_begin_of_new_word`）从上面这个 commit 导入，不复制改写
- 我们对官方脚本的修改放在本仓库 `proj/` 下，不改安装包里的代码
- 依赖版本：搭好环境后执行 `pip freeze > proj/requirements.lock`，三个人统一使用。至少包括 `transformers`、`tokenizers`、`tiktoken`、`jieba`、`torch`
- API key 只放环境变量，不进 Git，也不写进数据文件

### ρ 的定义（与 proposal 一致）
- ρ = 保留**词数**的比例：每个 passage 保留得分最高的 ⌈ρn⌉ 个词，按原顺序输出
- 主比较中的 M0、M1、M2、M4 和随机 baseline 都用我们自己的推理（见 B2），因此词预算完全相同
- 官方 `PromptCompressor(rate=ρ)` 只用于"原版 M0"的诊断。它的 `rate` 是目标参数，实际保留率不保证等于 ρ
- 报告中另外列出实际的 cl100k token 保留率和 reader-token 保留率，只作描述，不设硬性阈值

## B. 待 pilot 验证的实现（G2 测试覆盖）

### B1. 切词：所有语言共用一个函数
```
segment(text, lang):
    zh: 用 jieba 切词并插入空格，记录每个插入空格的位置
    en / sw: 原文不变
    用 XLM-R tokenizer 编码（带 offset_mapping）
    按官方 is_begin_of_new_word 规则（force_tokens=[]，token_map 默认）把 token 归到词
    返回：input_ids、每个 token 对应的 word_id、每个词在原文中的字符区间
```
- 英语也用这套规则，不用官方的 spaCy lemma，这样 M4 与 M1/M2 的标签流程相同
- 训练直接使用保存下来的 `input_ids` 和 `word_ids` 展开标签（`segment.expand_labels`），**不对词重新 tokenize**
- 实现：`proj/segment.py`；测试：`proj/test_segment.py`
  - jieba：在相邻的两个非空白 jieba token 之间插入一个空格，`inserted_spaces` 记录插入位置；去掉这些空格就是原文
  - 单独一个 `▁` token 的 offset 指向的是**下一个字符**（tokenizer 的行为），所以它不给词区间贡献字符
  - **只有空白的词**：一个 `▁` 后面紧跟着另一个新词的开头（例如标点）时，会产生这种词。它的区间为空，和官方推理一样算作一个词。aligner 给它的标签总是 0；pilot 中 zh 有 492 个，en 有 18 个，sw 有 4 个
- **拒绝映射不干净的样本**：XLM-R 内部会做 NFKC 规范化，字符位置可能因此丢字或重叠（例如 `"Café"` 的组合重音不属于任何词；`"㈱"` 被展开成 `"(株)"`，三个词落在同一个字符上）。`segment()` 每次都检查"有序、不重叠、覆盖所有非空白字符"，不满足就抛出 `SegmentError`；调用方把样本记为 `segment_error` 并保留原因，不生成标签。pilot 的 120 条里没有出现这种情况
- 已验证（120 条 pilot 样本）：词区间有序、不重叠、不含空白；原文每个非空白字符恰好属于一个词；与官方推理的合并结果逐词一致（比较前做 NFKC 规范化；词表里没有的字在官方输出中是 `<unk>`，这类词只比较词数）；标签展开到子词正确；按这些词做纯删除，aligner 全部判为 `ok`

### B2. 推理（主实验）
1. `segment()` 得到 token 和词
2. 模型前向，得到每个 token 的 p(keep)，按 word_id 取平均，得到词得分
3. 保留得分最高的 ⌈ρn⌉ 个词。分数相同时取位置靠前的词，保证结果确定。**只有空白的词不计入 n，也永远不保留**（它们不包含任何字符，留着只会占用预算）
   - 输入用 XLM-R 自己的 `<s>` … `</s>`，和训练时一样。官方推理用的 `TokenClfDataset` 没有传 `model_name`，默认按 BERT 处理，首尾放的是 `[CLS]`/`[SEP]`，在 XLM-R 词表里都是 `<unk>`（id 3，已实测）。这只影响"原版 M0"的诊断，不影响主比较
4. 从原文字符区间重建文本：原文中相邻的两个保留词，保留原来的间隔；不相邻时，en/sw 插入一个空格，zh 不插入。jieba 插入的空格不会出现在输出里
5. 随机 baseline：同样的 n 和 ⌈ρn⌉，随机种子固定并记录

### B3. Aligner（`proj/align.py`，以原文的词和字符位置为基准）
- 接口：`align(text, word_spans, teacher_text) -> {labels, status, reason}`
- 匹配：两边去掉全部空白后，找一个**完整原词**按原顺序组成的子序列，拼起来要等于 teacher 输出。不对 teacher 输出切词
- 边界检查（防止字符相同但词边界被改了）：
  - teacher 的空白落在某个保留词内部 → 不合法（例如 "a nice" → "an ice"）
  - 两个保留词在原文中间有空白、teacher 没有空白，且两侧都是字母或数字 → 不合法（例如 "a part" → "apart"）；一侧是标点则允许（例如 "rafiki yangu." → "rafiki."）
  - teacher 在词边界处多加空白 → 允许
  - 边界规则在**搜索过程中**就生效：在所有合法对齐里找最靠前的那个（例如 "a a,b" → "ab"，跳过第一个 "a"，取第二个）
  - 只有"词被合并"才能按原顺序对上时，返回 `uncertain`，不报 `reordered`
- 状态：
  - `ok`：合法的纯删除，输出 labels
  - `added`：输出中有原文没有的字符（新增或改写）
  - `reordered`：能证明输出由原词组成（每个词的使用次数不超过原文），但顺序不同
  - `uncertain`：不能证明属于以上任何一类（只删了词的一部分、词被合并或拆开、空输出、搜索超出步数上限）
  - 只有 `ok` 有 labels；其他状态 labels 为 `None`，样本不进入训练
- 歧义：有多种对齐方式时（例如重复词），取最靠前的位置，`reason` 写 `ambiguous`。统计时按语言报告歧义样本的比例
- `teacher_stats()`：统计量不依赖对 teacher 输出切词
  - 官方 `label_word.py` 的做法是把 teacher 输出切成词再比较。换成我们的切词后，单位会对不上：原词 `["天","气"]`、teacher 输出被重新切成 `["天气"]`，明明完整保留，却得到 variation_rate = 1、alignment_gap = −1。所以**不采用官方的这两个公式**
  - `char_variation_rate`：teacher 输出中原文没有的字符（按字符多重集）所占比例
  - `char_comp_rate`：teacher 字符数 / 原文字符数（都不计空白）
  - `word_keep_rate`：保留词数 / **有效词数**，只对 `ok` 样本计算。有效词由 `align.content_words()` 判定（区间非空），B2 推理的预算 n 也必须用这个函数
  - 对齐是否成功由 `align()` 的状态表示，不再用 alignment_gap 这个数字
- 验证：pilot 全部 120 个 chunk，每个分别按 20% / 50% / 90% 的比例随机删词，全部判为 `ok`，保留词拼起来等于输出

### B4. 数据格式（每个 chunk 一行 JSONL）
```
id, lang, source_doc_id, split,
text,                  # 原文
prep_text,             # 送入 XLM-R 的文本（zh 是插入空格后的版本）
inserted_spaces,       # zh 插入空格的位置；其他语言为 []
word_spans,            # 每个词在原文中的字符区间
input_ids, word_ids,   # 特殊 token 的 word_id 为 null
labels,                # 每个词一个标签
teacher: {model, prompt_version, params, raw_output, usage, status}
```
- teacher 的原始输出永远保留；重新对齐时不需要再次付费调用

### B5. Chunk（`proj/sample_wiki.py`）
- 在句末符号之后断句（`.`、`\n`，zh 再加 `。！？`），按顺序把整句装进 chunk。每个 chunk 同时满足：tiktoken `gpt-4` ≤ 512，**且** XLM-R（原文，不计特殊 token）≤ 510
  - 和官方不同：官方按 token 切再 decode，中文会把一个字切进两个 token；而且只限制 gpt-4 token，pilot 中 40 个英语 chunk 里有 24 个超过了 XLM-R 510
- 例外：单句本身超出上限时，按字符切开
- 检查（脚本内 assert）：拼回去等于原文；ID 唯一；没有空 chunk；train/dev 的文档不重叠
- zh 的 XLM-R 长度按 **jieba 插入空格之后**计算，和 `segment()` 看到的一致。按原文计算时，pilot 中 zh 最长到了 543
- 另外，`test_segment.py` 对每个 chunk 检查 `segment()` 的长度 ≤ 510
- Pilot：每种语言 30 train / 10 dev，每篇文档只取第一个 chunk。manifest 中记录了版本、方法和三个文件的 SHA-256；同一 seed 重跑，SHA-256 完全一致（已验证）
- 原文保持原样，不做人工清洗（zh 中有去掉模板后留下的残缺片段，例如"（，约），和"）；质量问题只记录
- **zh 繁转简**（已决定）：处理顺序是 原始文档 → OpenCC `t2s` → 切 chunk → `segment()` → teacher → 对齐
  - train 和 dev 使用同一规则；OpenCC 版本（`opencc==1.4.2`）和配置都记录在 manifest 中
  - `text` 和所有字符位置都以**转换后的文本**为准；`text_orig` 保留同一段的原始字符（只在转换不改变长度时才能一一对应，否则为 `null`，manifest 中统计了数量）
  - Pilot：40 个 zh chunk 里有 26 个被转换，没有一个改变了长度；zh 已重新生成，哈希已更新

### B6. Teacher（`proj/run_teacher.py`）
- 本地模型（学校没有提供 API）。三种语言使用同一个模型、同一个 prompt、同一组参数
- 第一个候选：`CohereLabs/tiny-aya-global`（3.35B，revision `af89d219b53ed9b13b8a4645f9c8028973510324`）
  - 官方语言列表包含 en、zh、sw；和 reader 候选 Qwen 不是同一系列；上下文 8K
  - gated 模型：要先在 HF 页面接受条款，再 `huggingface-cli login`。许可证是 CC-BY-NC-4.0（非商业），课程项目可以用
  - 备选：`CohereLabs/tiny-aya-earth`（revision `0624fafbe200f930b89566fc4ec4da314ba48d89`），侧重非洲语言。如果 Global 的 sw 表现差，用**同一批**样本测 Earth
  - 不用 Aya Expanse 8B：它的官方语言列表里没有 Swahili
- prompt：官方 `compression_instructions.json` 第 4 条**原样使用**（`PROMPT_VERSION = llmlingua2-instr4-verbatim`），包括里面写死的 "you (GPT-4)"。只有发现问题（例如把 zh/sw 翻译成英语）才出 v2，并在变更记录里写明
- 生成：BF16，`do_sample=False`（greedy decoding，可复现），`max_new_tokens=1024`。不再使用 `temperature=0.3`
- 每条保存：完整原始输出、结束原因（`eos` / `length`，`length` 表示被截断）、输入和输出的 token 数、耗时、显存峰值、模型 revision、各包版本、GPU 型号
- 输出不做任何后处理（不去引号、不去前缀），直接交给 aligner；格式问题会体现在状态里
- **能否进入训练**：`usable = (status == "ok") and (finish == "eos")`。被截断的输出即使对齐合法，也不能用
- **续跑保护**：每条记录保存 `teacher.config`（模型、revision、prompt 版本及其 SHA-256、dtype、生成参数）和输入文本的 SHA-256。续跑时，配置或输入文本只要有一处不同就直接报错，要求换 `--out`，绝不混用旧结果
- **半行 JSON**：只允许修复文件**最后一行**——截掉写了一半的记录；如果是最后的换行符丢了，就补上。中间行损坏一律报错。测试见 `proj/test_run_teacher.py`
- **运行环境（CSIL）**：代码在 `~/713proj`；venv 和 HF 缓存放在 `/local-scratch/dfa54`（本机磁盘，NFS home 已用 84%）。Python 3.12，`torch==2.15.0+cu132`（cu130 源里没有 2.15.0），其他依赖按 `requirements.lock`。运行前设置 `HF_HOME=/local-scratch/dfa54/hf HF_TOKEN_PATH=~/.cache/huggingface/token`。注意 `/local-scratch` 只在这台机器上（secb1010-b12）
- 评估（初筛，不能直接定下全量用哪个 teacher）：先每种语言 3 条跑通，再扩到 10 条。看四件事：
  1. 删除是否合法（`ok` 比例和失败原因）
  2. 删了多少（`char_comp_rate`、`word_keep_rate`）。原文原样返回也是 `ok`，所以必须看这一项
  3. 核心信息有没有保住（人工看）
  4. 有没有被截断
  - 每种语言 10 条时，一条结果就会让比例变 10 个百分点，所以只适合初筛
- **prompt v2**（`llmlingua2-instr4-outputonly-1shot`）= v1 的指令和 5 条规则 + 明确的输出要求（只输出结果，不要解释、Markdown、重建文本，保持原语言）+ 1 个**与输入同语言**的纯删除示例。三种语言用同一句内容（自己编写，与 Belebele 和 pilot 都无关；测试验证了三个示例都是合法删除）。sw 示例没有请母语者核对
- 结果文件按 prompt 版本分开存放：`results/teacher/{model}__{v1|v2}.jsonl`，v1 的结果完整保留

### B6 pilot 结果：Tiny Aya Global（2026-10-30，secb1010-b12，同一批 9 条 train，每种语言 3 条）

| | v1（官方 prompt 原样） | v2 |
|---|---|---|
| usable / ok | 0/9 | 0/9 |
| 截断 | 1（en，跑满 1024 token，还附带了"Reconstructed Text"） | 0 |
| Markdown / 开场白 | 8/9 | 0/9 |
| char_comp_rate 平均（en / zh / sw） | 0.91 / 2.42 / 0.50 | 0.48 / 0.62 / 0.37 |
| 原文没有的字符占比 | 2%–79% | 0.2%–8% |
| 速度 / 显存峰值 | 约 58 tok/s / 6.5 GB | 同左 |

- 诊断（不进入正式流程）：v1 去掉 Markdown 和开场白后仍然 0/9；v2 连标点也忽略后仍然 0/9。剩下的都是真正的改写：句首改成大写，中文改字（"北京矿务局"→"北京矿局"，"1523年至1533年"→"1523-1533年"），繁简混用（"汝贤"→"汝賢"），sw 换词（"hadi"→"na"）
- 结论：v2 解决了格式和截断问题，输出大体上已经是删词了，但**仍然没有一条是严格合法的删除**。单靠 prompt 不够，需要组里决定下一步（见 C）

### B7. 约束解码原型（`proj/constrained.py`，`--decoding constrained`；**原型，尚未替换正式流程**）
- 机制：生成时只允许输出完整的原词，按原顺序，每个位置最多用一次。规则在字节层面实现，因为 teacher 的 tokenizer 是 byte-level BPE，一个汉字可能被拆进多个 token
  - 词与词之间：可以不加空格，也可以加一个空格。原文中间有空白、且两侧都是字母或数字时，必须加空格（和 aligner 的边界规则一致）
  - 只有写完一个完整的词之后，才允许输出 `<|END_RESPONSE|>`；之后只允许 `<EOS_TOKEN>`。空输出不可能出现
  - 模型决定保留哪些词、何时结束，约束只屏蔽不合法的 token。输出仍然交给 aligner 独立验证
  - 异常不做修补，只记录：`no_legal_path`、`length`（截断）、`segment_error`（不生成）
- 配置指纹里加了 `decoding = deletion-grammar-v1` 和 `constrained.py` 的 SHA-256；结果写入单独的文件 `{model}__{prompt}__constrained.jsonl`
- 测试（`test_constrained.py`，用玩具字节词表）：随机生成的每一条约束输出都被 aligner 判为合法；反过来，合法的删除输出都能被约束接受；另外覆盖了各条边界规则
- 测试过程中发现并修复了 **aligner 的一个 bug**：它把"只含空白的词"当成了原文里有空白。在中文里，这类词来自 jieba 插入的空格，原文里并没有空白。v1/v2 的结论不受影响（那 18 条失败都是因为多出了原文没有的字符）
- **措辞**：三种语言（包括 M4 对照组）采用相同的机制，结论只限于"这种约束 teacher 生成的监督"。"3B 模型就够用"目前只是待验证的假设

### B7 结果：v2 + 约束解码（2026-10-30，同一批 9 条）
- 实现层面：9/9 合法（aligner 独立判为 `ok`），0 截断，0 空输出，0 `no_legal_path`，没有照抄。速度 43–53 tok/s（自由生成约 58），显存 6.5 GB
- 质量：词保留率 3%–44%（多数 7%–9%）。最长连续跳过的词数占原文 16%–92%：sw 有两条是 61%–62%，zh 有一条是 92%。有的输出结尾是拼凑的数字（"volkano,200 275,000 30"），有的只剩"索菲亚，波兰天主教徒"。我自己读下来约 4–5/9 可以接受，**这不是正式评估，sw 没有经过母语者核对**
- 合法率只能说明实现正确，**不能说明 teacher 选词有用**。长距离删除本身可能是合理的，真正的问题是丢了关键信息，或者变成了数字拼接
- **开发样本**：这 9 条（每种语言 pilot train 的前 3 条）已经经过多轮调试。方案确定后，必须在**没参与过调整的新样本**上验证（例如每种语言 pilot train 的第 4–10 条），才能决定是否扩大生成
- **v3**（`llmlingua2-instr4-outputonly-1shot-target33`）= v2 + 一句 "Keep about one third (33%) of the original words."。v2 的示例没有改，它本身保留了约 65% 的词，与目标不一致，这是已知的干扰因素
- **v3 结果**（约束解码，同样 9 条，`report_teacher.py` 对比）：**没有改善**。6/9 的输出与 v2 逐字相同；另外 3 条压得更狠、内容更差（sw_19469 只剩 "Ghana, kata Mbeya Mjini"；zh_7276488 后半段变成了词语堆砌）。实际保留率与 33% 目标的平均偏差：v2 0.213，v3 0.229。数字拼接和"索菲亚，波兰天主教徒"都没变。速度相同，9/9 合法
- 结论：给目标保留率改变不了 Tiny Aya 的选词。按事先定好的顺序，下一步是**换更大的 teacher**（仍用约束解码和 prompt v2，以便和本轮对比）
### B8. 更大的 teacher（2026-10-30/31，secb1010-b12；约束解码 + prompt v2 固定不变，每次只换模型）
- 选候选的条件：官方资料明确列出 en、zh、sw；量化后能装进 24 GB；最多比较两个
  - `CohereLabs/aya-101`（revision `e7dad472b9de8a30a00cf08a05c19003bf59028d`）：12.9B，mT5 encoder–decoder，Apache-2.0，不是 gated。模型卡的语言表里有 eng、swa、zho。权重是 fp32（48 GB），bf16 也要约 26 GB，所以用 bitsandbytes 8-bit（bf16 计算）
  - `Qwen/Qwen3-14B`（revision `40c069824f4251a91eefaf281ebe4c544efd3e18`）：14.8B dense，Apache-2.0。官方博客 qwenlm.github.io/blog/qwen3 的语言表里列有 Chinese 和 Swahili（共 119 种）。和 reader 候选 Qwen2.5 同一系列：reader 固定、所有压缩模型走同一套评测流程即可，但报告里要注明。用 bitsandbytes 8-bit，`enable_thinking=False`
  - 没选：Gemma 3（人工审核的 gated 模型，模型卡没有明确的语言列表）；Aya Expanse（不含 Swahili）
- **适配 aya-101**（不只是改 TokenTable）：
  - `TokenTable.from_hf` 能识别 SentencePiece（Metaspace decoder，`▁` 表示空格，**没有 byte fallback**）
  - 约束只在 token 边界上判断"剩下的部分能否被 token 恰好拼出"（这是充分条件），避免开始一个写不完的词而走进死胡同。mT5 会把全角"，"规范化成","，所以中文的全角标点是不可达的词，只能删掉；每条记录里有 `unreachable_words`
  - SentencePiece 的第一个 token 通常是 " word"，所以加了参数 `allow_leading_space`，只对 SentencePiece 开启；Tiny Aya 的规则不变
  - encoder–decoder 生成：prompt 是纯文本 = SYSTEM + 空行 + user prompt，不用 chat template；decoder 的 prompt 长度按 1 算；结束 token 就是 `</s>`；decode 时 `clean_up_tokenization_spaces=False`
  - 测试：玩具 SentencePiece 词表（没有单字节 token、去掉了全角标点）和**真实的 aya-101 词表**上做随机约束生成，都不会走进死胡同，并且每条输出都合法
- **aya-101 结果**（int8，9 条）：9/9 合法，0 截断。速度约 14 tok/s，显存峰值 16.4 GB。但它**基本不做有选择的删词**：
  - 4/9 原样照抄（保留率 100%：en_1300690、en_43621503、sw_19469、sw_2007）
  - 3/9 只保留开头 20%–25% 就结束（en_25196216、sw_16522、zh_3584166）
  - zh 的另外两条保留率 79%–81%，基本是原文去掉全角标点——这是 mT5 词表造成的，不是 teacher 的选择
  - 保留下来的内容事实准确（比 Tiny Aya 的数字拼接好），但作为压缩 teacher 没有明显改善
- **Qwen3-14B 结果**（int8，v2，9 条）：9/9 合法，0 截断。速度约 8 tok/s，显存峰值 16.2 GB
  - **确实在删词**，en 的删法接近 LLMLingua 的风格：去掉冠词和虚词（"Stone of Jacob appears in Book of Genesis as stone used as pillow…"）；en_25196216 保留 19.7%，像电报一样但信息很密；zh_3584166 保留 27%，要点在
  - 但**压得不够**：平均保留率 en 0.68 / zh 0.65 / sw 0.94；sw 有 2/3 接近照抄，zh 的两条在 77%–92%
  - 没有出现 Tiny Aya 的数字拼接，也没有 aya-101 的"只保留开头"
- 实际配置：两个模型都用 bitsandbytes 8-bit、bf16 计算、batch 1，`device_map={"": 0}`，24 GB 显存够用，不需要调整量化方式
- **Qwen3-14B + v3**（目标保留率 33%；只改 prompt 这一个因素）：9/9 合法，速度和显存不变
  - en 和 zh 压得更多了：平均保留率 en 0.68 → 0.45，zh 0.65 → 0.41；与目标的平均偏差 0.469 → 0.366
  - en_1300690：84% → 15.5%，保住了主线（Stone of Jacob、Genesis、Bet-El、vision、consecrated、Bethel = House of God）
  - zh_7275461：92% → 25%，但子女列表被压成了"1528年，1532年，丹麦王后 挪威王后…"这样的碎片，**丢了结构**（原文本身就缺人名）
  - en_43621503 和 sw_16522 仍然原样照抄（100%）；**sw 平均 0.93，基本没有压缩**
- GPU 时间（从 00:59 开始计时）：aya-101 约 3 分钟，Qwen3 v2 约 6 分钟，Qwen3 v3 约 5 分钟，合计约 15 分钟，远低于 2 小时的预算。中间 VPN 断过一次，重连后 `secb1010-b12` 直连时 DNS 解析失败，改为通过 `csil-cpu10` 跳转（`ssh -J dfa54@csil-cpu10.csil.sfu.ca:24 -p 24 …`）。机器没有重启，`/local-scratch` 完好

### B8 对比总表（同一批 9 条开发样本，约束解码，都是 9/9 合法）

| teacher | prompt | 保留率 en / zh / sw | 与 33% 的偏差 | 照抄 | 主要问题 | tok/s | 显存 |
|---|---|---|---|---|---|---|---|
| Tiny Aya Global 3.35B（bf16） | v2 | 0.13 / 0.22 / 0.08 | 0.213 | 0 | 过度压缩、跳得太远、数字拼接，丢关键信息 | 48 | 6.5 GB |
| Tiny Aya Global | v3 | 0.10 / 0.13 / 0.07 | 0.229 | 0 | 同上，忽略长度目标 | 47 | 6.5 GB |
| aya-101 12.9B（int8） | v2 | 0.74 / 0.62 / 0.73 | 0.438 | 4 | 照抄或只保留开头，不做有选择的删词 | 14 | 16.4 GB |
| Qwen3-14B（int8） | v2 | 0.68 / 0.65 / 0.94 | 0.469 | 3 | 删词方式合理，但压得太少 | 8 | 16.2 GB |
| **Qwen3-14B（int8）** | **v3** | **0.45 / 0.41 / 0.93** | **0.366** | 3 | en/zh 明显变好；**sw 基本不压缩**；zh 有一条丢了结构 | 8 | 16.2 GB |

（保留率是每种语言 `ok` 样本 `word_keep_rate` 的平均值；sw 的质量没有经过母语者核对。）

### 当前决定（2026-10-31）
**选定 Qwen3-14B int8 继续验证；英文和中文初步可用，Swahili 尚未通过 teacher 质量检查。** 暂停换模型，也不批准全量生成。

- **更正**：之前把问题归因于"小模型的选词能力"，以及猜测"模型对 Swahili 没把握"，都下得太早。Qwen3 官方明确把 Swahili 列为覆盖语言（预训练涵盖 119 种语言和方言），但接触过一种语言，不代表擅长在这种语言里做纯删除压缩。现在能说的只有：**当前的模型 + prompt + 约束机制，在这 3 条 Swahili 上基本照抄**。还分不清是理解能力、示例、任务指令还是解码方式造成的
- **为什么 sw 必须处理**：如果约 92% 的标签都是"保留"，M2 学到的主要是"少删词"。到时即使 M2 没有提升，也不能解释成"目标语言监督没用"，因为监督本身就没有展示有效的压缩。也**不能为了压到 33% 而强迫它随便删词**

### B9. 最后一轮有边界的诊断：v4（示例改为保留约三分之一）
- 只改一个因素：v3 的三个示例都只删了约三分之一（保留 0.61–0.68），和"保留约 33%"的目标冲突。v4 把三个示例的压缩结果改成保留约三分之一，其他（模型、量化、约束、指令文字）都不变
  - en："library built in 1890, closed 2015."（保留 0.30）
  - zh："图书馆1890年建造，2015年关闭"（0.36）
  - sw："Maktaba ilijengwa mwaka 1890, ilifungwa mwaka 2015."（0.33）——**没有经过母语者核对**
  - 三个都是合法删除（测试覆盖），内容一致：library / built / 1890 / closed / 2015
- **判定标准（跑之前定好）**：在同样的 9 条开发样本上，sw 算"明显改善"需要同时满足：
  1. sw 平均保留率 ≤ 0.60
  2. 3 条 sw 里最多 1 条照抄（保留率 ≥ 0.95）
  3. 人工看下来，sw 输出保住了主要事实（主语 / 是什么 / 数字 / 地点）。sw 仍然注明"非母语者判断"
  - en 和 zh 不能明显变差：平均保留率不能比 v3 回升超过 0.15，也不能出现新的丢结构或数字拼接
- **下一步按结果走**：
  - 明显改善 → 固定配置，在**新样本**上验证（每种语言 pilot train 的第 4–10 条，共 21 条，以前没参与过任何调整），重点看重要事实是否保留，而不只是看长度
  - 仍无改善 → **暂停全量 Swahili 标注**，和组员、老师讨论别的监督来源，或者把"当前 Swahili teacher 失败"作为可行性结果写进报告，并缩减 Core。不再继续调参
- 速度优化（例如 nf4）放到质量过关之后再做，以免同时改变输出质量

### B9 结果（Qwen3-14B int8，约束解码，v4，同样 9 条开发样本）：**没有通过**
| 标准 | v3 → v4 | 是否通过 |
|---|---|---|
| sw 平均保留率 ≤ 0.60 | 0.93 → 0.62 | ✗（差一点） |
| sw 照抄最多 1 条 | 2 → 1（sw_16522，0.97） | ✓ |
| sw 保住主要事实 | sw_19469（0.63）事实都在；sw_2007（0.26）前半段不错，结尾却变成乱语 "Tsunami huko kwa kubwa na kuwa na 30." | 部分（非母语者判断） |
| en 回升 ≤ 0.15 | 0.45 → 0.71（+0.26）：en_25196216 从 0.19 变成照抄 0.996 | ✗ |
| zh 回升 ≤ 0.15 | 0.41 → 0.59（+0.18）：zh_7276488 从 0.71 变成 0.98；zh_7275461 从 0.25 变成 0.58，子女列表的结构恢复了一部分 | ✗ |

- 9/9 合法，0 截断，速度和显存不变
- **观察**：每条样本基本只有两种状态，要么照抄（≥ 0.95），要么压缩（0.14–0.63）。prompt 稍微一改，就有样本在两种状态之间翻转（v3→v4：en_25196216 从压缩变成照抄，sw_2007 从照抄变成压缩）。三种语言都有，不只是 sw
- 一个**还没验证的假设**（不作为结论）：在 greedy + 约束解码下，如果开头几步选了"下一个原词"，后面就容易一直照抄下去。也就是说，问题可能出在解码方式，而不只是 prompt 或语言能力
- **按 B9 事先定好的规则执行：暂停全量 Swahili 标注，不再调参。** 也不开始新样本验证，因为配置没有通过开发样本这一关
- 要和组员、老师讨论的选项：
  1. 换一种 sw 的监督来源（例如人工标注少量 sw 样本，或者先翻译、在英语上压缩、再对齐回来——每种都会改变实验含义，需要重新论证）
  2. 把"当前的本地 teacher 在 Swahili 上做不出合格的删除标签"作为可行性结果写进报告，缩减 Core（例如先做 en/zh，M2 改为 stretch）
  3. 如果组里认为值得，**只做一次**针对解码方式的诊断（例如检查照抄是不是从前几步就开始的），而不是继续改 prompt
- en/zh 也**不能**直接进入全量：同样存在照抄和压缩之间的翻转，需要在新样本上看照抄比例是否稳定

### 范围决定（2026-11-03）：缩减 Core，sw 移到 stretch
这是对 B9 第 2 个选项的决定，取代 `plan.md` 和 proposal 中相应的 Core 范围。
- **Core**：M0、M1（zh）、M4（en control）。M1 和 M4 的 teacher、数据来源类型和训练预算相同
- **H1**：只检验 M1 是否提升中文表现，即 acc_zh(M1) − acc_zh(M0)
- **H2**：只限中文，检验中文监督的收益是否超过等预算的英文监督，即 [acc_zh(M1) − acc_zh(M0)] − [acc_zh(M4) − acc_zh(M0)]。结论只支持中文这个案例，不能推广为低资源语言普遍有效
- **Stretch**：sw 全量标注、M2。如果保留 sw 上的 M0/M1/M4 评测，标为探索性迁移分析；sw reader 不通过不再阻塞 en/zh Core
- **报告中的可行性结果**：在当前本地计算预算及已测试的模型、prompt 和约束解码配置下，Swahili 开发样本未通过预设的 teacher 质量检查，因此未启动全量标注，并将 M2 移至 stretch。该结果基于反复用于开发的 3 条 Swahili 样本，人工质量判断未经母语者核验，不能据此认定 Swahili 的删除式监督不可行
- **前提没变**：en/zh 仍要通过未参与调参的新样本验证，才能全量生成。缩减 sw 只解决了范围问题，没有解决 en/zh 的 teacher 稳定性问题（见 B9 的照抄和压缩翻转）

### en/zh 新样本验证方案（2026-11-03 已固定，开跑前 commit）
- 配置：Qwen3-14B，固定现有 revision、int8、prompt v3、greedy + 约束解码、`max_new_tokens=1024`；本轮不同时改量化或解码。选择 v3 的依据是已有开发样本，不使用本轮输出选 prompt
- 样本：按各语言 pilot JSONL 的文件顺序，先筛选 train，再取第 4–10 条，en/zh 各 7 条。本地已有 teacher 结果中没有这些 ID；这只能核实没有已保存的生成结果，不能证明没有其他人工接触
  - en：`en_66883368_c0`, `en_55626531_c0`, `en_35796523_c0`, `en_674_c0`, `en_27016797_c0`, `en_55622294_c0`, `en_55620821_c0`
  - zh：`zh_2864757_c0`, `zh_5553_c0`, `zh_2858674_c0`, `zh_1026130_c0`, `zh_3581312_c0`, `zh_2865097_c0`, `zh_5644091_c0`
- 通过标准：**每种语言分别满足全部条件**，不能用 en 的成功抵消 zh 的失败
  1. 7/7 条 `status == ok` 且 `finish == eos`，无空输出和 `no_legal_path`。每种语言 6/7 只有 85.7%，所以“至少九成”在这里等价于 7/7
  2. 照抄（`word_keep_rate >= 0.95`）最多 2/7 条
  3. 7 条 `word_keep_rate` 的算术平均值 ≤ 0.60，使用 B3 的有效词分母；不能先排除失败样本再宣告通过
  4. 人工逐条核对原文和输出，7/7 均保留主要事实。先只看原文，记录该 chunk 的主体、核心事件或陈述及关键限定，再看输出；出现主体或关系混淆、关键否定丢失、关键数字与对象错配、丢失主旨或不可理解的拼接，均判失败，记录理由。允许删除次要事实，不要求保留每项细节。中文由能读中文的审核者判断
- 任何条件失败就暂停扩大生成，回组里讨论；不根据这 14 条继续选 prompt、改阈值或换样本凑通过。基础设施故障可记录原因后以相同配置续跑，模型输出失败不能当作基础设施故障重抽
- 使用独立结果文件 `results/teacher/Qwen3-14B__v3__constrained__new_en_zh.jsonl`，保留原始输出和配置指纹。逐条人工判断存放在 `results/teacher/Qwen3-14B__v3__constrained__new_en_zh_review.csv`，字段为 `id, main_facts, preserved, failure_reason, reviewer`。CSIL 上从代码目录运行（环境变量见 B6）：
  ```bash
  python run_teacher.py --model Qwen/Qwen3-14B --langs en,zh --split train --start 3 --n 7 --prompt v3 --decoding constrained --quant int8 --max-new-tokens 1024 --out results/teacher/Qwen3-14B__v3__constrained__new_en_zh.jsonl
  ```
- 这是能否扩大 teacher 验证的可行性检查，不是总体质量估计，也不替代 G5 中文 100 条人工 audit、成本与速度检查和其他全量前 Gate。尚未运行，不得写成已通过

### B8 结论与推荐
- **推荐的候选**：Qwen3-14B + 约束解码 + prompt v3。它是唯一一个既真正在删词、又能按长度目标调整的 teacher。**但还不能扩大试验**，原因有三：
  1. **sw 基本不压缩**（0.93），而 sw 是这个项目的关键语言。这一点不解决，M2 的监督就近似"全保留"
  2. **还没有在新样本上验证**：这 9 条是开发样本，必须在每种语言 pilot train 的第 4–10 条上重新验证
  3. **速度**：int8 只有约 8 tok/s。按每个 chunk 约 300 个输出 token 估算，3 种语言 × 约 3k chunk 在一张 4090 上大约要 100 小时。全量之前需要先测更快的方案（例如 nf4，或者其他推理框架），并重新比较质量
- **还没解决的问题**：sw 照抄的原因（可能是 v2 示例本身只删了约 35%，削弱了长度目标；也可能是模型对 Swahili 没把握，只敢保守地全留）；zh 的列表类内容压缩后丢结构；mT5 词表不能输出全角标点（只影响 aya-101）；Qwen3 和 reader 候选 Qwen2.5 同一系列，报告里要注明
- 原计划（组里已定）：只改一个变量——在 prompt 里给出目标保留率（v3），仍用约束解码，在同样的 9 条上比较。如果没有改善，再试更大的 teacher。限制跳跃距离留作以后的消融实验，暂时不和其他规则叠加。注意：官方 aligner 的窗口是用来**找匹配位置**的，而这里的窗口会**限制 teacher 能选什么内容**，两者作用不同

### G2 必须覆盖的测试
- 训练和推理对同一文本得到相同的 `input_ids` 和词边界
- 每个词的标签正确展开到它的所有子词；特殊 token 和 padding 不错位
- 纯删除的 teacher 输出：保留词重建的文本 = teacher 输出（空白规范化之后）
- 重复词、标点、换行、完全不删、新增词、换序、改写的手写例子
- zh：jieba 插入的空格不出现在交给 reader 的文本里；原文自带的空格保留

### B10. 污染过滤（G3，`proj/contamination.py`；测试：`proj/test_contamination.py`）
- 来源：`facebook/belebele`，revision `7899cdfa4e1e0d733fd77c848e2c273cb1d32be2`，字段 `flores_passage`，每种语言 488 个不重复 passage
- Normalization：NFKC → casefold → 去掉 Unicode 类别 P、S、Z、C 开头的字符（标点、符号、空白、控制和格式字符）；字母、combining mark、数字保留
- n-gram：字符级，en/sw 50，zh 20。归一化后最短的 passage 为 en 115、zh 42、sw 149，所以最短 passage 的一半也至少有一个完整 n-gram
- 判定：和**同语言**的 passage 有至少 1 个相同 n-gram 就删除整个 chunk，不设比例阈值。一半 passage 拼进 512 token 的 chunk 时只占很小比例，比例阈值会漏掉
- 位置：OpenCC `t2s` 之后、切分之前。zh 的索引只有简体，所以输入必须已经转换。split 由文档 ID 的哈希决定，删除 chunk 不影响其他 chunk 的 split
- 局限：只能查出逐字复制（忽略标点、大小写和空白）。查不出改写、翻译或同主题的内容。一个 chunk 里不足 n 个字符的 passage 片段也不会被删
- 结果：pilot 120 个 chunk 删除 0 个（`test_contamination.py` 把这批真实文本固定为负例）
- 文档抽样（`scan_contamination.py`，证据在 `results/contamination/wiki_scan.json`，含文档 ID、匹配片段和对应 passage）：每种语言 3000 篇完整文档，en 0 篇命中，zh 1 篇，sw 1 篇。0.03% 是这个样本的**文档命中率**，样本只有 2 个命中，不能当作误删率，也不等于 chunk 的删除率
  - 人工读了两篇命中的上下文和 passage（非正式核对），都只共享一个机构名，内容无关：zh_1310（英国大学列表）与一篇讲苏格兰司法的 passage 共享“邓迪大学（University of Dundee）”；sw_4035（Gertrude Mongella）与一篇讲阿姆斯特丹的 passage 共享 UNESCO 的 Swahili 全称
  - 所以这两篇都不是污染，而是会被当前规则误删。目前不为此调高 n；全量后要抽看被删样本，再决定是否调整
- 开发过程（本地记录，repo 里没有可复跑的脚本）：实现者只拿到 spec，没有看测试；初版 38 个测试一次通过。之后 18 个 mutation 有 3 个存活，补了 5 个测试用例，实现没有改动。pilot 负例测试是再之后加的
- **这组参数是实现时选的默认值，待组员确认**

## C. 待决定

- [ ] Teacher 最终用哪个模型：先试 Tiny Aya Global（见 B6），看初筛结果再定
- [x] ~~Tiny Aya 自由生成 0/9 合法，下一步选约束解码还是换更大的模型~~ → 两者都做了，见 B7、B8
- [ ] 是否采用 Qwen3-14B + 约束解码 + v3 作为 teacher（B8 推荐，但有 3 个前提没满足）
- [x] ~~sw 不压缩怎么办~~ → 做了 B9，没有通过；已决定缩减 Core，sw 全量标注和 M2 移到 stretch（见“范围决定”）
- [ ] 全量生成的速度方案（int8 只有约 8 tok/s）
- [ ] 候选 reader 及其尝试顺序（G1 之前固定）
- [ ] 训练时特殊 token 和 padding 的标签处理：沿用官方训练脚本，还是用 -100 忽略
- [ ] 确认污染过滤的 normalization、n-gram 大小和阈值（G3，默认值见 B10）
- [ ] 是否按 `char_comp_rate` 的极端值过滤（例如 teacher 几乎什么都没删，或者几乎全删了），看 pilot 的分布后再定

## C2. 质量过滤（已决定，偏离 proposal）

proposal 写的是"按 variation rate 和 alignment gap 各删最差的 10%"。我们改成：
1. **先排除所有非 `ok` 样本**，按状态和原因统计数量，写进报告
2. `ok` 样本的 `char_variation_rate` 恒为 0，alignment gap 由状态代替，所以**不按百分位删除**：指标全是 0 就全部保留，不为了凑比例删掉合格数据
3. 如果以后加入有意义的连续指标（见上面 `char_comp_rate` 一项），阈值事先固定并记录，平局一律保留（用 `>`，不用 `>=`）
4. 非 `ok` 样本不参与任何百分位计算

原因：
- 官方 `filter.py` 是依次过滤，并且用 `>=` 第 90 百分位。如果超过 90% 的样本是 0，阈值就是 0，所有样本都会被删掉
- 官方的两个指标依赖对 teacher 输出切词，单位对不上（见 B3）
- 严格的 aligner 已经在样本层面排除了不合法的标签

## D. 尚未核实的外部说法

- Lukauskas et al. 2026 是否使用官方 `rate`、能否与我们的数字直接对比：未看过其代码，不作为结论

## 变更记录

| 日期 | 改了什么 | 原因 | 影响的结果 |
|---|---|---|---|
| 2026-10-30 | 初版 | — | — |
| 2026-10-30 | B5：切块加 XLM-R ≤ 510 上限，改为按句子打包 | 英语超长会被静默截断；官方的按 token 切会破坏中文字符 | pilot 样本（已重新生成） |
| 2026-10-30 | B3：aligner 改为"完整原词子序列 + 边界检查"，状态改为 ok/added/reordered/uncertain；加 `teacher_stats()` | review 发现字符子序列会放过 "a nice" → "an ice" 这类改写 | 还没有真实标签，不影响已有结果 |
| 2026-10-30 | A：加 `proj/requirements.lock`（pip freeze） | 复现 | — |
| 2026-10-30 | B5：zh 先做 OpenCC t2s 再切 chunk，加 `text_orig` | Belebele 是 `zho_Hans` | zh pilot 已重新生成（新哈希见 manifest） |
| 2026-10-30 | B3：边界规则放进搜索过程（找合法的最早对齐）；`teacher_stats()` 改为字符级 | "a a,b" → "ab" 被误判；官方指标在我们的切词下单位对不上 | 还没有真实标签 |
| 2026-10-30 | C2：质量过滤改为"排除非 ok，全零不删"，偏离 proposal | 见 C2 | 需要在 plan.md 和报告中说明 |
| 2026-10-30 | B1：实现 `segment()`；B2：只有空白的词不计入 n，用 `<s>`/`</s>` | 见 B1、B2 | — |
| 2026-10-30 | B5：zh 切块长度按 jieba 插入空格后计算；字段 `n_xlmr_tokens_raw` 改名为 `n_xlmr_tokens` | 按原文计算时 zh 最长 543，超过 510 | pilot 重新生成：zh 的切块变了；en/sw 文本不变，但因字段改名哈希也变了 |
| 2026-10-30 | A：venv 加入 torch（CPU）、jieba、固定 commit 的 llmlingua；更新 `requirements.lock` | `segment()` 需要 | 4090 上要换成 CUDA 版 torch |
| 2026-10-30 | B1：`segment()` 拒绝映射不干净的样本（`SegmentError`）；B3：`word_keep_rate` 只用有效词做分母 | review 构造出了丢字和重叠的例子；只有空白的词让 zh 的保留率偏低 | — |
| 2026-10-30 | B6：teacher 改为本地 Tiny Aya Global，greedy decoding，prompt 原样使用；去掉 temperature | 学校没有提供 API | — |
| 2026-10-30 | B6：加 `usable`（被截断的不能用）、续跑时校验配置和输入文本、修复写了一半的最后一行 | review | — |
| 2026-10-30 | B6：加 prompt v2；结果文件按 prompt 版本命名（v1 文件改名为 `__v1`，内容不变）；汇总里加 `near_copy` | v1 是 0/9，且 8/9 带 Markdown | v1 结果保留 |
| 2026-10-30 | B7：约束解码原型；B3：aligner 不再把只含空白的词当作原文有空白 | v2 仍是 0/9；约束解码的测试暴露了 aligner 的 bug | v1/v2 的状态不变 |
| 2026-10-30 | B7：加 prompt v3（目标保留率 33%）和 `report_teacher.py`；测试改为检查所有已有结果的配置指纹 | 检验过度压缩是不是因为缺少长度目标 | 无改善，见 B7 |
| 2026-10-31 | B8：约束解码支持 SentencePiece 和 encoder–decoder（aya-101），支持 Qwen3-14B（`<\|im_end\|>`、关闭 thinking）、bitsandbytes 量化；`unreachable_words` | 评估更大的 teacher | 新增 3 个结果文件，旧结果的配置指纹不变（测试覆盖） |
| 2026-10-31 | 当前决定：选定 Qwen3-14B int8 继续验证，sw 没通过 teacher 质量检查；B9：prompt v4（示例改为保留约三分之一）、`--start` 参数；更正之前的归因 | review | B9 没有通过 → 暂停 sw 全量标注 |
| 2026-11-01 | 更正舍入：Tiny Aya v2 zh 0.21→0.22，Qwen3 v3 sw 0.92→0.93（按原始 JSONL 重算） | 外部 review | 只是数字更正，结论不变 |
| 2026-11-03 | B10：加污染过滤（G3）和默认参数 | G3 | pilot 没有 chunk 被删，已有结果不变 |
| 2026-11-03 | 范围决定：Core 改为 M0、M1、M4，H1/H2 限于中文，sw 全量标注和 M2 移到 stretch | B9 没有通过，sw 监督质量不足 | 所有 sw 和 M2 计划；已有结果不变 |
| 2026-11-03 | B10：更正“误判”的表述，区分文档命中率和误删率；加 `scan_contamination.py`、命中证据和 pilot 负例测试 | 外部 review | 过滤行为不变 |
| 2026-11-03 | 加 en/zh 新样本验证方案（配置、14 个 ID、按语言的判定规则、人工判断文件）；将工作区 `plan.md` 原样同步到仓库；补命中证据的来源与许可说明 | 开跑前明确标准，让组员能获取计划，核实 Belebele CC BY-SA 4.0 | 尚未生成新结果；验证标准已在开跑前固定 |
