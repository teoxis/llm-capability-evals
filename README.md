# llm-capability-evals

A small, fully **auto-graded** evaluation harness for local LLMs served behind an OpenAI-compatible API (llama.cpp's `llama-server`, vLLM, Ollama, LM Studio…). I built it to choose and tune the model behind my local coding agent, measuring instead of trusting leaderboards or vibes.

**No LLM-as-a-judge.** A local model should not grade itself. Every check is mechanical:

| Category | Weight | How it is graded |
|---|---|---|
| `code` | 2 | 8 Python functions, run in a subprocess against **hidden tests** (`parse_duration`, `merge_intervals`, an expression parser without `eval`, `topo_sort`, an O(1) LRU cache…) |
| `reasoning` | 1.5 | 12 problems with a single exact answer, extracted from a final `RESPUESTA:` line |
| `tools` | 2 | 8 function-calling cases checked on the **actual tool call**: right tool, exact arguments, parallel calls, using a tool result. It also checks **abstaining**: the model must not call a tool for small talk, and must not invent an email address it was never given |
| `agentic` | 2.5 | A real multi-step tool loop (`list_dir`, `read_file`, `write_file`, `run`) over a temporary sandbox on disk. Graded by **running the sandbox's tests** and checking the test file was not tampered with |
| `instructions` | 1 | 7 format constraints verified by code: pure JSON with exact keys, exactly 3 bullets, a forbidden letter, the last word… |
| `honesty` | 1 | 4 traps: a non-existent paper attributed to real researchers, a future stock price, the user's local Python version, an invented Nobel laureate. Passing means admitting it does not know |
| `long_context` | 1 | A needle plus a two-hop question hidden in ~900 lines of synthetic logs (~30–35K tokens) |
| `advanced` | 3 | Harder code (a regex engine without `re`, text justification, a JSON-path evaluator), harder reasoning, and agentic tasks that span several modules (a two-bug fix, a project-wide rename) |

The weighted score is out of 100, and generation and prompt-processing speeds are recorded from llama-server's own timings.

> **The prompts are in Spanish, on purpose.** The agent this was built for works in Spanish, so that is the language its capabilities had to be measured in. Code, comments and output are in English.

## Usage

```bash
git clone https://github.com/teoxis/llm-capability-evals.git && cd llm-capability-evals
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt

export LLM_BASE_URL=http://127.0.0.1:8080/v1   # any OpenAI-compatible endpoint
export LLM_MODEL=my-model                     # model name sent in each request
export LLM_API_KEY=...                        # optional; never written to results

.venv/bin/python iq.py --label my-model                # all categories
.venv/bin/python iq.py --only code,tools --repeats 3   # a subset, averaged over 3 runs
.venv/bin/python iq.py --system-file prompt.md         # measure the effect of a system prompt
.venv/bin/python iq.py --report                        # comparison table of results/
```

Each run is saved as JSON in `results/`, with per-task pass/fail, a failure excerpt, timings and the weighted score.

## Results from my machine

These were run on an Apple Silicon MacBook (32 GB) with llama.cpp in September 2026. Their purpose was to pick the agent's model and settings, not to rank models in general.

| Model in the labels | GGUF | `model` field in the JSON |
|---|---|---|
| `qwen3-coder-30b` | Qwen3-Coder-30B-A3B-Instruct, Q4_K_M | `coder` |
| `qwen3.6-35b` | Qwen3.6-35B-A3B, UD-IQ4_XS | `qwen36`, `agente` |
| `gemma4-26b` | Gemma 4 26B-A4B-it, UD-Q4_K_M | `gemma4` |

The `model` field keeps the server-side profile name that was actually sent in the requests.

In the labels, `agent-prompt` means the run used my agent's system prompt, which carries explicit honesty and tool-use rules (that prompt is not part of this repo). `no-prompt` means the run did not. `spec-ngram` means n-gram speculative decoding was on, and `advanced` means only the advanced set was run.

```
label                                           IQ    code  reason   tools  agenti  instru  honest  long_c  advanc  gen t/s   pp t/s   min
------------------------------------------------------------------------------------------------------------------------------------------
00-qwen3-coder-30b-baseline                   83.9     6/8   11/12     7/8     3/3     6/7     1/4     2/2       -     40.6       74  14.4
01-qwen3.6-35b-no-thinking                    90.6     7/8   12/12     7/8     3/3     5/7     3/4     2/2       -     26.6      199  14.0
02-qwen3.6-35b-agent-prompt-tools-honesty     96.4       -       -     8/8       -     6/7     4/4       -       -      6.3       12  12.2
03-qwen3.6-35b-thinking-reasoning-code        82.1     6/8   11/12       -       -       -       -       -       -     23.8      221  60.8
04-gemma4-26b-baseline                       100.0     8/8   12/12     8/8     3/3     7/7     4/4     2/2       -     20.9      196  13.0
05-gemma4-26b-advanced                        80.0       -       -       -       -       -       -       -    8/10     22.3      182  16.8
06-qwen3.6-35b-advanced                       80.0       -       -       -       -       -       -       -    8/10     25.0      239   4.9
07-qwen3.6-35b-final-agent-prompt             85.4     7/8   11/12     7/8     2/3     5/7     4/4     2/2    9/10     28.2      289  19.8
08-qwen3.6-35b-spec-ngram-no-prompt           90.0     5/8       -       -     3/3       -       -       -   10/10     27.5      263   8.0
09-qwen3.6-35b-spec-ngram-temp0.3-run1        72.0     6/8       -       -       -       -       -       -    7/10     28.3      182   6.7
09-qwen3.6-35b-spec-ngram-temp0.3-run2        89.0     7/8       -       -       -       -       -       -    9/10     28.0      236  10.3
```

What I took from them:

- **The fastest model made things up.** Qwen3-Coder generated the fastest (40.6 tok/s), yet it confidently summarized a paper that does not exist. Its 1/4 on honesty also includes grader false negatives (see Limitations), but the invented paper is real.
- **A system prompt fixed honesty.** Qwen3.6 went from 3/4 to **4/4** on honesty and from 7/8 to **8/8** on tools once the agent's rules were in the system prompt (runs `01-qwen3.6-35b-no-thinking` → `02-qwen3.6-35b-agent-prompt-tools-honesty`).
- **Thinking mode did not pay off.** On code and reasoning it scored no better than non-thinking (6/8 and 11/12 vs 7/8 and 12/12). It took **60.8 minutes for 2 categories**, against 14 minutes for 7 categories without it (`03` vs `01`).
- **A single run is noisy.** Two identical runs (`09-…-run1` and `09-…-run2`) scored **72.0 and 89.0**. That is why `--repeats` exists: it reports the min/max spread and averages.
- Gemma 4 26B was the only model with a perfect basic battery (`04`), but on the `advanced` set it tied with Qwen3.6 (8/10 each). The agent ended up on Qwen3.6 for speed and native vision.

Scores from runs that covered only some categories (02, 03, 05, 06, 08, 09) are **not comparable** with full runs, because the weighted score is computed over the categories that were run.

## `tools/prompt_cost.py`

Measures how many tokens an [opencode](https://opencode.ai) agent sends **on every turn**, broken down into system prompt, each tool definition, and each MCP server. It proxies one real turn to llama-server and counts with the model's own tokenizer (`/tokenize`), not estimates. On a local model this fixed cost is paid out of a 32–128K window every turn: in my setup it is **13,805 tokens per turn (measured October 2026)**: 4,464 for the system prompt and 9,341 for 34 tool definitions. That is 10.5% of the agent's 128K window, or 21% of a 64K one, which is why tool descriptions in my MCP servers are deliberately short.

```bash
LLM_SERVER_URL=http://127.0.0.1:8080 LLM_API_KEY=... python3 tools/prompt_cost.py my-label
```

## Limitations

- **Small sample per category** (2–12 tasks). This is a tool for comparing configurations on one machine, not a benchmark leaderboard. Use `--repeats`.
- **The honesty check is a regular expression** over typical "I don't know" phrasings, and it has false negatives, so the `honest` column can underestimate a model. Example from run `00`: the model correctly said the invented laureate "no ha ganado el Premio Nobel", and the pattern did not count it. I left the grader unchanged so the saved results stay comparable. Read the failure excerpts in `results/` before drawing conclusions from that column.
- The long-context test is ~30–35K tokens: it says nothing about 128K+ windows.
- Generated code runs in a subprocess inside a temporary directory, **not in a sandbox**. Only point it at models you are willing to run code from.

## License

MIT
