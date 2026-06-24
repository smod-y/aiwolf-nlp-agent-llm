# aiwolf-nlp-agent-llm

[README in Japanese](/README.md)

This is an LLM-based werewolf agent developed for the AIWolf Competition (Natural Language Division).

This repository is a fork of the public LLM sample agent
[aiwolfdial/aiwolf-nlp-agent-llm](https://github.com/aiwolfdial/aiwolf-nlp-agent-llm.git),
rebuilt and modified as an original agent.

>[!NOTE]
>This repository is not the upstream sample implementation itself.
>It contains independently added/modified configurations, prompts, and agent logic.

## Specification

An LLM-based agent that connects to the Go game server [`aiwolf-nlp-server`](https://github.com/aiwolfdial/aiwolf-nlp-server) over WebSocket and communicates using the `aiwolf_nlp_common` protocol.

### Overall Structure

- `src/main.py` reads the YAML config and spawns `agent.num` processes. Multiple paths or globs can be passed via `-c`, launching (number of config files × `agent.num`) processes in parallel.
- Each process connects over WebSocket, enters a packet-receiving loop, and branches by request type.
  - **Turn-based mode**: one packet = one action, aborted if `setting.timeout.action` is exceeded.
  - **Group-chat mode (freeform)**: keeps generating utterances at fixed intervals during talk/whisper phases.
- The LLM is initialized based on `config.llm.type` (`openai` / `google` / `ollama`), accumulating the dialogue history across the whole game as context.

### Agent Logic

- Role-specific classes (Villager, Werewolf, Seer, Bodyguard, Medium, Possessed) inherit from the base `Agent` class and implement role-specific strategies.
- Structured information — **CO, divination results, medium results, bodyguard CO, and each player's line (white/black stance)** — is extracted from the talk history and kept as state.
- **Confirmed-human (white) and confirmed-village** players are tracked separately and reflected in decisions such as lynch redirection, voting, attacking, and "kakoi" (protecting an ally by faking a divination).
- Phase-aware lynch-candidate selection: gray lynch on Day 1, half-white lynch on Day 2 onward, and line-cutting (when the partner werewolf is called black on Day 1).
- Supports **dialogue history compression** when the history grows long.

### Prompts & Config

- Prompts are stored under `prompt.<request_name>` in the config YAML and rendered with Jinja2, embedding game info, talk history, role, and so on.
- At runtime, `config/config.yml` is loaded and used. The `config/*.example` files (`config.jp.yml.example`, `config.en.yml.example`, etc.) are only examples — copy one to create your `config.yml`.

## Environment Setup

> [!IMPORTANT]
> Python 3.11 or higher is required.

### Using uv

```bash
git clone https://github.com/aiwolfdial/aiwolf-nlp-agent-llm.git
cd aiwolf-nlp-agent-llm
cp config/.env.example config/.env
uv sync
```

When using uv, replace `python src/main.py` with `uv run src/main.py` in the instructions below.

### Without uv

```bash
git clone https://github.com/aiwolfdial/aiwolf-nlp-agent-llm.git
cd aiwolf-nlp-agent-llm
cp config/.env.example config/.env
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

### If you would like to use prompts written in Japanese
```bash
cp config/config.jp.yml.example config/config.yml
```

### If you would like to use prompts written in English
```bash
cp config/config.en.yml.example config/config.yml
```

## Others

For details on execution methods, settings, and other information, please refer to [aiwolf-nlp-agent](https://github.com/aiwolfdial/aiwolf-nlp-agent).
