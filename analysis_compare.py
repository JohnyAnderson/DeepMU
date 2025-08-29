import os
import json
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

# === 模型配置 ===
BASE_MODEL = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
ADAPTER_PATHS = [
    "../../lora-deepseek_7B_qwen",
    "../../lora-deepseek_7B_adapter_terminology",
    "../../lora-deepseek_7B_bug_only"
]
BATCH_SIZE = 4
FLASH_ATTENTION = True

def load_model():
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=quant_config,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="flash_attention_2" if FLASH_ATTENTION else "eager"
    )
    for path in ADAPTER_PATHS:
        model = PeftModel.from_pretrained(model, path)
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    model.eval()
    return tokenizer, model

def build_batch_prompt(batch):
    prompts = []
    for item in batch:
        bug_description = item["bug_description"].strip()
        mymodel_output = item.get("mymodel_output", "").strip()
        # Prompt 编写，明确任务
        prompt = f"""
你是一个经验丰富的智能合约审计专家。请根据以下规则判断 mymodel_output 中的 bug 描述是否覆盖了 bug_description 中提到的内容，并输出唯一的结果。

【任务规则】

* 如果 bug_description 为 none，若 mymodel_output 意思是该合约是健康合约，不包含任何漏洞，则输出：correct；否则输出：wrong
* 如果 mymodel_output 中所提及的漏洞与 bug_description 中所有 bug 类型直接或间接关联，则输出：correct
* 如果 mymodel_output 涉及了 bug_description 中部分 bug 类型，则输出：partly correct
* 如果 mymodel_output 中的漏洞与 bug_description 中的任何 bug 类型都无关，则输出：wrong
* 只要有"间接影响"、"可能诱发"、"实现依赖中可能出现"等等级别的弱关联，也认为是"涉及"。

【TP / FP / FN 判定规则】

你需要在上述判断的基础上，进一步对模型表现进行如下分类，输出 "TP"、"FP" 或 "FN" 中的一个：

* **TP（True Positive）**
    - 当 `bug_description` 不为空，且 `mymodel_output` 能完整识别出所有列出的 bug 类型；
    - 或者，当 `bug_description` 为空，且 `mymodel_output` 明确说明合约中没有漏洞。

* **FP（False Positive）**
    - 当 `bug_description` 为空，模型却输出了某种漏洞描述（即错误地判断合约有 bug）。

* **FN（False Negative）**
    - 当 `bug_description` 不为空，模型却未提及任何相关漏洞（即输出表示没有漏洞或输出为空）；
    - 或者仅部分识别出 `bug_description` 中的漏洞类型。

【强化逻辑映射规则】

* 只要合约功能涉及 **链上多个模块之间状态依赖或同步操作**（如交易路径、滑点控制、套利同步），就应考虑其是否存在 **跨组件或跨系统状态不一致风险**，即视为 `cross bridge inconsistency`；
* 如果合约设计使用多个链上组件之间的读写过程（如 router 操作多个 token pair、oracle 读取历史价格窗口），这些组件状态不同步即可构成"逻辑桥不一致"；
* 若套利函数未在一次交易中完成全部更新，或者执行顺序对最终状态有影响，这就是 `atomicity violation`；
* 若套利、闪电贷等造成的交易后状态未被回滚或写入成功前中断，可能导致 `inconsistent state update`；
* 即使没有出现"跨链桥"这个字样，只要有"跨资产路径操作 + 状态传播 + 同步风险"，都应认定为 `cross bridge inconsistency`。
* 对于涉及协议版本迁移的合约（如V1到V2迁移器），其跨版本状态同步特性本身就构成cross bridge inconsistency风险
* 任何涉及多个独立组件（如factory、router、pair）状态同步的操作，都应视为潜在的cross bridge inconsistency
* 对于router类合约，其核心功能就是处理多资产路径交换，这种跨资产操作本身就隐含cross bridge inconsistency风险
* 迁移类合约中任何未完全原子化的状态转移操作，都应视为cross bridge inconsistency
* 如果合约需要协调多个独立组件的状态（如同时操作V1和V2的流动性池），即使在同一链上，也构成cross bridge inconsistency

【强化联想规则】
请你从语义、机制、后果和攻击路径多个维度进行推理判断，不要局限于关键词匹配。

下面这些行为即使没有直说，也应视为"涵盖相关漏洞",并且你的推断逻辑中只要出现"有可能涉及"就认为涉及，并作为结果处理：

* 对于router、oracle、bridge等中间件合约，其跨组件操作特性本身就构成cross bridge inconsistency风险
* interface合约中未经验证的外部调用或状态依赖，可能隐含privilege escalation风险
* math/library合约中的套利计算、价格同步等逻辑，其状态依赖特性可能隐含cross bridge inconsistency
* 如果存在 **时间操控/时间敏感操作**，并影响合约的**权限逻辑、操作有效期判断、定时调用控制**，就与 `privilege escalation` 存在潜在关联；
* 如果有 **重入攻击、部分写入失败、状态未提交、算术溢出/下溢** 等问题，即使未明说，也属于 `atomicity violation` 或 `inconsistent state update`；
* 如果合约涉及资产流通路径（如 token → ETH → pair → pool），即使未说明是"跨链桥"，只要有中间状态风险、顺序同步问题、swap 路径混乱等，都可认定为 `cross bridge inconsistency`；
* 如果是 interface、router、oracle 等基础设施类合约，虽然本身不执行逻辑，但其结构、上下游依赖、授权操作或未加验证流程中出现风险，同样构成 `privilege escalation` 或 `business logic flaw` 的表现；
* **溢出/精度/定价误差** 所导致的状态写入不准确，也可视为 `state update inconsistency`；
* 若 mymodel_output 提到"transaction context 失效"、"gas 不足导致迁移失败"等问题，也可能导致原子性问题或链间同步失败。
【补充推理规则】
- 接口（interface）或抽象合约虽无实现，但其授权结构、调用要求、状态返回逻辑可能导致权限越权或状态异常，需审视其上下游影响，亦视为涉及相关漏洞。
- 任意进行资产路径计算、套利策略推导、滑点控制、价格调优等逻辑的库类合约，若依赖外部合约状态或不能确保计算中间状态的一致性，均视为 cross bridge inconsistency。
- 凡价格预言机类合约，其数据源依赖外部池子或时间窗口更新，均构成跨组件状态同步要求，应识别为跨模块状态同步风险。
- 若合约设计中包含 fallback、pending 状态缓冲、gas失败容错逻辑等，表示其试图缓解状态写入失败风险，应视为与 inconsistent state update 强相关。
- 若合约通过时间戳控制用户操作权限、资金提取条件或锁仓解锁逻辑，该控制路径即使合理，也存在潜在的 privilege escalation 或 price manipulation 风险。

【漏洞类型定义】

1. **Privilege Escalation（特权升级）**：权限验证缺失或逻辑漏洞，攻击者能越权调用敏感函数；
2. **Inconsistent State Update（状态更新不一致）**：多步骤操作未保持一致性，部分执行导致脏数据或状态异常；
3. **Atomicity Violation（原子性违规）**：本应原子执行的操作被中断或部分完成；
4. **Cross-Bridge Inconsistency（跨链桥不一致）**：不仅限于跨链场景，任何需要多个独立组件/模块/版本间状态同步的操作，如果存在状态不一致风险，都属此类
5. **Business Logic Flaw（业务逻辑漏洞）**：功能实现违背预期业务模型，造成资金逻辑错误或流程断裂；
6. **Price Manipulation（价格操纵）**：定价机制依赖可控数据源，导致可被恶意利用获取不当利益。

【输出格式要求】

你只需要输出以下两行，不要输出任何额外解释内容：

result: correct / partly correct / wrong （三选一，全部小写）
verdict: TP / FP / FN （三选一，大写）

示例输出：

result: correct  
verdict: TP

---

以下是内容:
bug_description:
{bug_description}

mymodel_output:
{mymodel_output}
        """.strip()
        prompts.append(prompt)
    return prompts

def batch_generate(prompts, tokenizer, model):
    inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to(model.device)
    input_len = inputs.input_ids.shape[1]

    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask,
            do_sample=False,
            max_new_tokens=1024,
            eos_token_id=tokenizer.eos_token_id,
            return_dict_in_generate=True,
            output_scores=False
        )

    generated_tokens = outputs.sequences[:, input_len:]
    results = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
    return results

def extract_result_and_metric(text: str):
    result = "wrong"
    metric = "FN"
    lines = [line.strip().lower() for line in text.strip().splitlines() if line.strip()]
    for line in lines:
        if line.startswith("result:"):
            val = line.split(":", 1)[-1].strip()
            if val in ["correct", "partly correct", "wrong"]:
                result = val
        if line.startswith("verdict:"):
            val = line.split(":", 1)[-1].strip().upper()
            if val in ["TP", "FP", "FN"]:
                metric = val
    return result, metric

def evaluate_all(json_path, start_idx=0):
    tokenizer, model = load_model()
    base_path, _ = os.path.splitext(json_path)
    output_path = base_path + "_evaluated.json"

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if os.path.exists(output_path):
        with open(output_path, "r", encoding="utf-8") as f:
            saved = json.load(f)
        data[start_idx:start_idx+len(saved)] = saved

    total = len(data)
    stats = {"correct": 0, "partly correct": 0, "wrong": 0}
    metrics = {"TP": 0, "FP": 0, "FN": 0}

    with tqdm(total=total - start_idx, desc="Evaluating") as pbar:
        for i in range(start_idx, total, BATCH_SIZE):
            batch = data[i:i+BATCH_SIZE]
            prompts = build_batch_prompt(batch)

            try:
                outputs = batch_generate(prompts, tokenizer, model)
            except Exception as e:
                print(f"❌ 第{i}条开始的batch推理失败: {e}")
                continue

            for j, output in enumerate(outputs):
                idx = i + j
                result, metric = extract_result_and_metric(output)
                data[idx]["evaluation_result"] = result
                data[idx]["metric"] = metric
                data[idx]["local_model_output"] = output
                stats[result] += 1
                metrics[metric] += 1
                pbar.set_postfix({**stats, **metrics})
                pbar.update(1)

            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)

    print("\n✅ 评估完成")
    print("🔢 结果分布：")
    for k, v in stats.items():
        print(f"{k:<15}: {v}")
    print("📊 TP / FP / FN 统计：")
    for k, v in metrics.items():
        print(f"{k:<5}: {v}")
    print(f"📝 评估结果保存至: {output_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--json_path", type=str, default="contract_bug_model_output.json")
    parser.add_argument("--start_idx", type=int, default=0)
    args = parser.parse_args()

    evaluate_all(args.json_path, args.start_idx)
