import os
import platform
import sys
import time
import torch
from minivllm import LLM, SamplingParams
from transformers import AutoTokenizer


def main():
    t_start = time.perf_counter()
    # 环境检查：仅在 Windows 原生环境下演示单卡
    if platform.system() != "Windows":
        print("[提示] 此脚本面向 Windows 原生环境，但也可在 WSL/Linux 运行。")
    if not torch.cuda.is_available():
        print("[错误] 未检测到可用的 CUDA。请安装支持 CUDA 的 PyTorch 并确保 GPU 可用。")
        sys.exit(1)

    # 模型路径：优先使用仓库内的示例模型，其次使用用户主目录
    repo_default = os.path.join(os.path.dirname(__file__), "huggingface", "Qwen3-0.6B")
    path = repo_default if os.path.isdir(repo_default) else os.path.expanduser("~/huggingface/Qwen3-0.6B/")
    if not os.path.isdir(path):
        print(f"[错误] 模型目录不存在：{path}\n请将 'path' 修改为你本地的 Qwen3-0.6B 路径。")
        sys.exit(1)

    # 加载分词器（用于 chat 模板或普通文本），fast 失败则自动回退到 slow
    try:
        tokenizer = AutoTokenizer.from_pretrained(path, use_fast=True)
    except Exception as e:
        print(
            "[警告] Fast tokenizer 加载失败，切换为 use_fast=False。\n"
            "可能原因：tokenizers 与 tokenizer.json 版本不兼容。建议升级：\n"
            "  pip install --upgrade tokenizers>=0.15.2 transformers>=4.51.0\n"
            f"详细错误：{e}\n"
        )
        tokenizer = AutoTokenizer.from_pretrained(path, use_fast=False)

    # Windows 原生环境：仅支持单卡张量并行
    # 若报错提示后端/并行限制，请切换到 WSL2/Linux 或将 tensor_parallel_size 设为 1
    t_init_start = time.perf_counter()
    llm = LLM(path, enforce_eager=True, tensor_parallel_size=1)
    t_init_end = time.perf_counter()

    # 构造输入：尽量使用 chat 模板；若不可用则用普通字符串
    user_prompt = "你好，简单介绍一下你自己，并给出一个 1 到 20 的质数列表。"
    try:
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": user_prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
    except Exception:
        prompt = user_prompt

    # 采样参数：适合快速验证
    sampling_params = SamplingParams(temperature=0.7, max_tokens=1280)

    try:
        t_gen_start = time.perf_counter()
        outputs = llm.generate([prompt], sampling_params, use_tqdm=False)
        t_gen_end = time.perf_counter()
    except ImportError as e:
        # 依赖友好提示：transformers/Qwen3、flash-attn、triton
        print(
            "[错误] 运行失败，可能缺少依赖。\n"
            "- 请确保 transformers 版本支持 Qwen3Config（建议 >= 4.51.0）。\n"
            "- 请在 WSL2/Linux 环境安装 flash-attn、triton 以支持 GPU 内核。\n"
            f"详细错误：{e}"
        )
        sys.exit(1)

    print("\n=== 生成结果 ===")
    print(outputs[0]["text"]) 

    # 注打开预热warmup()生成时间比较久，但运行时间会比较快：
    # === 运行时间统计 ===
    # LLM 初始化耗时：751.782s
    # 文本生成耗时：35.542s
    # 脚本总耗时：787.952s
    # 运行时间统计
    t_end = time.perf_counter()
    print(
        "\n=== 运行时间统计 ===\n"
        f"LLM 初始化耗时：{t_init_end - t_init_start:.3f}s\n"
        f"文本生成耗时：{t_gen_end - t_gen_start:.3f}s\n"
        f"脚本总耗时：{t_end - t_start:.3f}s"
    )


if __name__ == "__main__":
    main()