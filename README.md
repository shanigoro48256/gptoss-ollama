# gptoss-ollama

## 概要
**gptoss-ollama** は、ローカル環境で **gpt-oss** を **Ollama** で実行するためのデモ用リポジトリです。  
以下の構成をサポートしています。

- **OpenAI SDK + Ollama** による gpt-oss 実行
- **LangChain + Ollama** による gpt-oss 実行
- **MCP + LangGraph エージェント** のデモ実行（ターミナルから実行）

---

## リポジトリのクローン

```bash
git clone https://github.com/shanigoro48256/gptoss-ollama.git
cd gptoss-ollama
````

---

## 環境変数の設定

`.env` ファイルを作成し、以下を設定してください。

```env
OPENAI_API_KEY=ollama
OPENAI_BASE_URL=http://ollama-runtime:11434/v1

# MCPサーバ用APIキー
TAVILY_API_KEY=＜Tavily Search APIキー＞   # Web検索
GITHUB_TOKEN=＜Githubキー＞               # Github検索
```

---

## Docker で環境構築

```bash
# コンテナ起動
docker compose up -d
```

---

## パッケージインストール
```bash
# コンテナへ入る
docker exec -it gptoss-ollama /bin/bash
```

```bash
uv pip install -e .
```

---

## モデルのダウンロード

新しいターミナルを開いて以下を実行してください。

```bash
docker exec -it ollama-runtime bash -lc "ollama pull gpt-oss:120b"
```

> 20B モデルを使用する場合は `gpt-oss:20b` を指定してください。

---

## デモの実行方法

### OpenAI SDK + Ollama、LangChain + Ollamaデモ（JupyterLabから実行）

* ブラウザで [http://localhost:8888](http://localhost:8888) を開く
* `src/infer.ipynb`を開き、コードセルを実行

### LangGraph + MCPエージェント デモ（ターミナルから実行）

```bash
cd src
python mcp_client.py
```

---

## ハードウェア要件

* **GPU:** NVIDIA A100 80GB推奨
* **gpt-oss:120b** VRAM 63GB
* **gpt-oss:20b** VRAM 14GB

---

## License

This project is licensed under the **MIT License**.
Copyright (c) 2025 shanigoro48256

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the “Software”), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

> The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
>
> THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
> IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

### サードパーティ製コンポーネントについて

本プロジェクトで使用する外部ライブラリ、モデル、フレームワーク等については、それぞれの元のライセンスに従ってご利用ください。
各コンポーネントのライセンス情報は、該当する公式のドキュメントをご参照ください。

---
