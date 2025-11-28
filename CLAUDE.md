# Project Context: NQS-guided SQD with FFNN + Transformer Samplers

You are collaborating with (蔡秀吉) on a quantum chemistry research project.

The goal is to study **sample-efficient Sample-based Quantum Diagonalization (SQD)**
using **Neural Quantum States (NQS)** as classical samplers, for small molecules
(starting from H₂) under **12–14-bit encodings**.

斷線恢復時，請先讀取 `docs/實驗記錄.md` 查看完整實驗歷程。

## 當前進度 (2024-11-28)

### 已完成

1. **VMC 化學精度達成**: ComplexFFNNNQS 達到 0.047 mHa (目標 1.6 mHa)
2. **SQD 整合完成**: 4-qubit H2 達到 0.000 mHa (精確 FCI)
3. **關鍵發現**: 實數 NQS 的 20.5 mHa 瓶頸源於符號問題，複數波函數解決此問題

### 下一步: 12-qubit H2 測試

目的: 4-qubit 系統太小 (僅 4 個物理配置)，無法展示 NQS 取樣優勢。需要更大系統。

執行步驟:
1. 使用現有 `build_h2_hamiltonian_12bit()` 建構 12-qubit Hamiltonian
2. 訓練 ComplexFFNNNQS (配置空間: 36 個 (1a,1b) 配置)
3. 比較 Random vs NQS 取樣的樣本效率
4. 指標: 達到化學精度所需的樣本數

預估時間: 5-15 分鐘

相關檔案:
- `test_nqs_sqd_integration.py`: 整合測試腳本
- `src/sqd_interface/sqd_runner.py`: SQD 執行器 (已修復積分格式問題)
- `src/sqd_interface/sampling_adapters.py`: 取樣器適配器 (含 ComplexFFNNSampler)

實作時請務必保持程式碼自然，不要讓人看出是 AI 生成。此外，實作也請務必遵循 Code Readability，記住代碼是寫給人看的，所以要有自然的代碼流程

### 關鍵參考檔案 (已驗證有效)

> 暫時權限密碼：Kinoshita15@

所有紀錄用的文檔都必須要精簡但保留所有必要的關鍵的資訊，遵循 TLDR，同時不可以有任何的 emoji。

> 此外整體專案都要有良好的檔案結構。

We focus on two ansatz families:

1. **FFNN-based NQS** (feed-forward neural networks) as a strong, simple baseline.
2. **Transformer-based NQS** (autoregressive / GPT-style) as a more expressive sampler,
   inspired by recent work on Transformer NQS and generative CI methods.

SQD is used as a **sample-based diagonalization backend**: the NQS produces bitstring
samples, SQD builds a subspace from these configurations, and then diagonalizes the
Hamiltonian in that subspace (with configuration recovery iterations). The core
question is:

> How much does a better sampler (FFNN vs Transformer NQS) reduce the **sample
> complexity** required to reach accurate ground-state energies (e.g. pushing from
> ~ -5.6 Ha toward ~ -7.63 Ha) for small molecular systems?


## Tech & Environment

- Language: Python 3.10+
- Libraries:
  - **Qiskit** + **`qiskit-addon-sqd`** for Sample-based Quantum Diagonalization
  - **PyTorch** for FFNN and Transformer NQS
  - NumPy / SciPy / Matplotlib / Pandas for analysis
- Hardware target: single NVIDIA RTX 4090 GPU on Linux / WSL2
- Workflow: **Claude Code (Opus 4.5)** + Skills + MCP tools

When editing instructions here, follow Anthropic's best practices for Claude 4.x and
Claude Code:

- Keep instructions concise and focused; iterate on this file like a prompt.
- Avoid overly aggressive language about tool usage (Opus 4.5 is very sensitive to
  system prompts).
- Use project memory (`CLAUDE.md`) for conventions, workflows, and key commands.

References:

- Claude Code settings & `CLAUDE.md`: https://docs.anthropic.com/en/docs/claude-code/settings
- Project memory: https://docs.anthropic.com/en/docs/claude-code/memory
- Claude 4.x / Opus 4.5 prompting best practices: https://docs.anthropic.com/en/docs/build-with-claude/prompt-engineering/claude-4-best-practices
- Claude Code agentic coding best practices: https://www.anthropic.com/engineering/claude-code-best-practices


## Directory Layout

- `src/nqs_models/`
  - `ffn_nqs.py`          : FFNN-based NQS models (real-valued log-ψ)
  - `transformer_nqs.py`  : GPT-style Transformer NQS (autoregressive sampler skeleton)
  - `utils.py`            : init, diagnostics, parameter counting
- `src/sqd_interface/`
  - `hamiltonian.py`      : molecular Hamiltonians (H₂, later H-chain) + 12-bit mappings
  - `sqd_runner.py`       : wrapper around `qiskit-addon-sqd`
  - `sampling_adapters.py`: adapters mapping NQS / baseline samplers → SQD input
- `src/experiments/`
  - `h2_12bit_ffn_baseline.py`        : FFNN NQS vs baseline sampler for H₂ @ 12-bit
  - `h2_12bit_transformer_sampler.py` : Transformer NQS sampler experiments
  - `ablation_nqs_vs_baseline.py`     : unified ablation driver
- `configs/`
  - YAML configs for molecules, encodings, sampler hyperparameters, and SQD options
- `skills/`
  - `nqs-sqd-research/`              : research skill for NQS+SQD (FFNN + Transformer)
  - `experiment-report-writer/`      : write structured experiment sections
- `.claude/commands/`
  - `nqs-theory-review.md`           : command for reviewing the theory & code
  - `plan-new-experiment.md`         : command for planning a concrete new run
- `.mcp.json`
  - MCP configuration for a shell-based server (and room to add python-exec later)
- `notebooks/`
  - sanity checks and derivations (Hamiltonian, mappings, toy experiments)
- `results/`
  - `raw/`, `processed/`, `figures/`


## Coding Standards

- Use **type hints** on public functions.
- Prefer clear, explicit code over clever one-liners.
- Document all non-trivial math / physics assumptions in docstrings or comments.
- Keep stochasticity explicit:
  - Always specify random seeds in experiments.
  - Distinguish between Monte Carlo noise, model bias, and SQD approximation error.

When editing numerical code, always propose a **sanity check** experiment (e.g. an
ultra-small toy Hamiltonian where you know the exact ground-state energy).


## Claude Code Workflow (Opus 4.5)

This repository is designed for agentic coding:

1. **Start from this folder in Claude Code.**
   - Claude will automatically load this `CLAUDE.md`, discover `skills/`, `.claude/commands/`,
     and `.mcp.json`.
2. **Use Skills for specialization.**
   - `nqs-sqd-research` for theory + algorithmic design.
   - `experiment-report-writer` for turning results into write-ups.
   - See: https://console.anthropic.com/docs/en/agents-and-tools/agent-skills/overview
3. **Use slash commands for repeatable tasks.**
   - `/nqs-theory-review` to summarize the current architecture & assumptions.
   - `/plan-new-experiment` to design the next experiment under 4090 constraints.
4. **Use MCP tools when appropriate.**
   - Shell MCP for safe file operations (`ls`, `cat`, `tail`).
   - Add a Python execution MCP later if you want to run experiments from within Claude
     (see https://modelcontextprotocol.io/docs/develop/connect-local-servers).

Follow Anthropic best practices for Claude Code:

- Start with a quick `/init` in new projects (already done here, but you can refine).
- Keep CLAUDE.md and Skills small and iterative—update them after you see how Opus 4.5 behaves.
- Use subagents / skills for specialized workflows instead of overloading one agent.
