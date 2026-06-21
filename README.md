# aiwolf-nlp-agent-llm

[README in English](/README.en.md)

人狼知能コンテスト（自然言語部門）向けに開発した、LLMを利用する人狼エージェントです。

本リポジトリは、公開されているLLMサンプルエージェント
[aiwolfdial/aiwolf-nlp-agent-llm](https://github.com/aiwolfdial/aiwolf-nlp-agent-llm.git)
をフォークし、独自のエージェントとして構築・改修したものです。

>[!NOTE]
>本リポジトリはフォーク元のサンプル実装そのものではありません。
>独自に追加・変更した設定、プロンプト、エージェントロジックなどが含まれています。

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
curl -LO https://github.com/aiwolfdial/aiwolf-nlp-server/releases/latest/download/aiwolf-nlp-server-linux-amd64
curl -LO https://github.com/aiwolfdial/aiwolf-nlp-server/releases/latest/download/default_5.yml
curl -LO https://github.com/aiwolfdial/aiwolf-nlp-server/releases/latest/download/default_9.yml
curl -LO https://github.com/aiwolfdial/aiwolf-nlp-server/releases/latest/download/freeform_5.yml
curl -Lo .env https://github.com/aiwolfdial/aiwolf-nlp-server/releases/latest/download/example.env
chmod u+x ./aiwolf-nlp-server-linux-amd64

## 実行方法
./server/aiwolf-nlp-server-linux-amd64 -c ./default_5.yml # 5人ゲームの場合
# ./aiwolf-nlp-server-linux-amd64 -c ./default_9.yml # 9人ゲームの場合
# ./aiwolf-nlp-server-linux-amd64 -c ./default_13.yml # 13人ゲームの場合
# ./aiwolf-nlp-server-linux-amd64 -c ./freeform.yml # チャット形式の場合

 uv run src/main.py -c config/config.yml # 実行方法
 uv run src/main.py -c config/config_5.yml