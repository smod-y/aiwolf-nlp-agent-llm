# aiwolf-nlp-agent-llm

[README in English](/README.en.md)

人狼知能コンテスト（自然言語部門）向けに開発した、LLMを利用する人狼エージェントです。

本リポジトリは、公開されているLLMサンプルエージェント
[aiwolfdial/aiwolf-nlp-agent-llm](https://github.com/aiwolfdial/aiwolf-nlp-agent-llm.git)
をフォークし、独自のエージェントとして構築・改修したものです。

>[!NOTE]
>本リポジトリはフォーク元のサンプル実装そのものではありません。
>独自に追加・変更した設定、プロンプト、エージェントロジックなどが含まれています。

## 仕様

Go 製ゲームサーバ [`aiwolf-nlp-server`](https://github.com/aiwolfdial/aiwolf-nlp-server) に WebSocket で接続し、`aiwolf_nlp_common` プロトコルでやり取りする LLM ベースのエージェントです。

### 全体構成

- `src/main.py` が YAML 設定を読み込み、`agent.num` 個のプロセスを spawn する。`-c` には複数パスや glob を指定でき、設定ファイル数 × `agent.num` のプロセスが並列起動される。
- 各プロセスは WebSocket に接続してパケット受信ループに入り、リクエスト種別に応じて処理を分岐する。
  - **ターン制方式**: 1 パケット = 1 アクション。`setting.timeout.action` を超えると打ち切る。
  - **グループチャット方式（freeform）**: トーク/囁きフェーズ中に一定間隔で発言を生成し続ける。
- LLM は `config.llm.type`（`openai` / `google` / `ollama`）に応じて初期化し、対話履歴をゲーム全体で累積させてコンテキストとする。

### エージェントロジック

- 役職別クラス（村人・人狼・占い師・狩人・霊媒師・狂人）が基底クラス `Agent` を継承し、役職固有の戦略を実装する。
- 発言履歴から **CO・占い結果・霊媒結果・狩人 CO・各プレイヤーのライン（白黒スタンス）** を構造化抽出し、状態として保持する。
- **確定白・確定村** を区別して管理し、吊り誘導・投票・襲撃・囲いなどの意思決定に反映する。
- グレー吊り（Day1）／片白吊り（Day2 以降）／ライン切り（Day1に相方人狼が黒出しされた場合）など、フェーズに応じた吊り候補選定方針を持つ。
- 履歴が長くなった際の **対話履歴の圧縮** に対応する。

### プロンプト・設定

- プロンプトは設定 YAML の `prompt.<request_name>` に格納され、Jinja2 でゲーム情報・発言履歴・役職などを埋め込んで生成する。
- 実際には `config/config.yml` を読み込んで使用する。`config/*.example`（`config.jp.yml.example` / `config.en.yml.example` など）はあくまで設定例であり、これをコピーして `config.yml` を作成する。

## 環境構築

> [!IMPORTANT]
> Python 3.11以上が必要です。

### uvを使用する場合

```bash
git clone https://github.com/aiwolfdial/aiwolf-nlp-agent-llm.git
cd aiwolf-nlp-agent-llm
cp config/.env.example config/.env
uv sync
```

uvを使用する場合、以下の`python src/main.py`などを`uv run src/main.py`と読み替えてください。

### uvを使用しない場合

```bash
git clone https://github.com/aiwolfdial/aiwolf-nlp-agent-llm.git
cd aiwolf-nlp-agent-llm
cp config/.env.example config/.env
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

### 日本語のプロンプトを使用したい場合
```bash
cp config/config.jp.yml.example config/config.yml
```

### 英語のプロンプトを使用したい場合
```bash
cp config/config.en.yml.example config/config.yml
```

## その他

実行方法や設定などその他については[aiwolf-nlp-agent](https://github.com/aiwolfdial/aiwolf-nlp-agent)をご確認ください。


## ダウンロード

```bash
curl -LO https://github.com/aiwolfdial/aiwolf-nlp-server/releases/latest/download/aiwolf-nlp-server-linux-amd64
curl -LO https://github.com/aiwolfdial/aiwolf-nlp-server/releases/latest/download/default_5.yml
curl -LO https://github.com/aiwolfdial/aiwolf-nlp-server/releases/latest/download/default_9.yml
curl -LO https://github.com/aiwolfdial/aiwolf-nlp-server/releases/latest/download/freeform_5.yml
curl -Lo .env https://github.com/aiwolfdial/aiwolf-nlp-server/releases/latest/download/example.env
chmod u+x ./aiwolf-nlp-server-linux-amd64
```

## 実行方法

サーバの起動:

```bash
./server/aiwolf-nlp-server-linux-amd64 -c ./default_5.yml # 5人ゲームの場合
./aiwolf-nlp-server-linux-amd64 -c ./default_9.yml # 9人ゲームの場合
./aiwolf-nlp-server-linux-amd64 -c ./freeform.yml # チャット形式の場合
```

エージェントの起動:

```bash
uv run src/main.py -c config/config.yml
```