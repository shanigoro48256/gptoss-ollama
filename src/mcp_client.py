import os
import sys
import json
import uuid
import asyncio
from contextlib import AsyncExitStack
from typing import Any, Dict, List, Tuple, Annotated, TypedDict
import httpx
from dotenv import load_dotenv
from langchain_ollama import ChatOllama
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, BaseMessage
from langchain_core.tools import StructuredTool
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph.message import add_messages
from langgraph.managed import RemainingSteps
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from pydantic import create_model, Field, BaseModel

# 環境変数読み込み
load_dotenv(override=True)

# MCPサーバ実行に使用するコマンド、対象のMCPサーバ
MCP_COMMAND   = "python"
MCP_SCRIPTS   = ["fs_mcp_server.py", "search_mcp_server.py"]
MCP_NAMESPACE = True

base_url = os.getenv("OPENAI_BASE_URL", "http://ollama-runtime:11434")
base_url = base_url[:-3] if base_url.endswith("/v1") else base_url
model_id = os.getenv("OLLAMA_MODEL", "gpt-oss:120b")

# 状態オブジェクトの型定義
class State(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    remaining_steps: RemainingSteps

# 会話スレッドの一意なID
def new_thread_id(user_id: str = "guest") -> str:
    return f"{user_id}-{uuid.uuid4()}"

# JSONスキーマ -> Python型への対応表
JSON_TYPE_MAP = {
    "string": str,
    "integer": int,
    "number": float,
    "boolean": bool,
    "array": list,
    "object": dict,
    None: str,
}

# JSONスキーマからPydanticモデルを生成（バリデーションの作成）
def _args_model_from_schema(name: str, schema: Dict[str, Any]) -> type[BaseModel]:
    props = (schema or {}).get("properties", {}) or {}
    required = set((schema or {}).get("required", []) or {})
    fields: Dict[str, Tuple[Any, Any]] = {}

    for key, spec in props.items():
        pytype = JSON_TYPE_MAP.get(spec.get("type"), str)
        desc = spec.get("description")
        default = spec.get("default", None)
        if key in required and default is None:
            fields[key] = (pytype, Field(..., description=desc))
        else:
            fields[key] = (pytype, Field(default, description=desc))

    if not fields:
        fields["payload"] = (dict, Field(default=None, description="free-form payload"))
    return create_model(f"{name}_Args", **fields)

# LLMやツール実行の戻り値を文字列に整形
def _stringify_call_result(result: Any) -> str:
    try:
        content = getattr(result, "content", None)
        if content is None:
            return json.dumps(result, ensure_ascii=False, default=str)
        out: List[str] = []
        for item in content:
            t_ = getattr(item, "type", None)
            if t_ == "text":
                out.append(getattr(item, "text", ""))
            elif t_ == "json":
                out.append(json.dumps(getattr(item, "json", None), ensure_ascii=False))
            else:
                out.append(f"[{t_}]")
        return "\n".join([x for x in out if x])
    except Exception:
        return str(result)

# MCPツール一覧をLangchainのToolに変換して登録
async def build_mcp_tools(session: ClientSession, label: str, namespace: bool) -> List[Any]:
    tools: List[Any] = []
    # MCPのツール一覧を取得
    resp = await session.list_tools()

    for t in resp.tools:
        base_name = t.name
        name = f"{label}.{base_name}" if namespace else base_name

        desc = t.description or f"MCP tool: {base_name}"
        schema = t.inputSchema or {"type": "object", "properties": {}}
        ArgsModel = _args_model_from_schema(name, schema)
        
        #LangchainからMCPツールを実行する関数
        def make_runner(tool_name: str = base_name):
            async def _run(**kwargs):
                result = await session.call_tool(tool_name, kwargs)
                return _stringify_call_result(result)
            return _run

        runner = make_runner()
        
        #LangChainのツールとして登録
        tool_obj = StructuredTool.from_function(
            name=name,
            description=desc,
            args_schema=ArgsModel,
            coroutine=runner,
        )
        tools.append(tool_obj)

    return tools

# LLM 初期化
def create_llm(model_id: str, base_url: str) -> BaseChatModel:
    return ChatOllama(
        model=model_id,
        base_url=base_url,
        temperature=0.1,
    )

def _forward_env() -> dict:
    env = os.environ.copy()
    for k in ["TAVILY_API_KEY", "GITHUB_TOKEN", "HTTP_PROXY", "HTTPS_PROXY", "NO_PROXY"]:
        if k in os.environ:
            env[k] = os.environ[k]
    return env

def _trunc(s, n=1000):
    s = s if isinstance(s, str) else json.dumps(s, ensure_ascii=False)
    return s if len(s) <= n else s[:n] + "…"

# プロンプトを入力して、エージェントを実行
async def run_agent(agent, user_input: str, config: dict):
    user_msg = HumanMessage(content=user_input)
    response = await agent.ainvoke(
        {"messages": [user_msg], "remaining_steps": 3},
        config=config,
    )
    if response and "messages" in response:
        msgs: List[BaseMessage] = response["messages"]
        last_user_idx = max((i for i, m in enumerate(msgs) if isinstance(m, HumanMessage)), default=-1)

        i = last_user_idx + 1
        while i < len(msgs):
            m = msgs[i]
            if isinstance(m, AIMessage) and getattr(m, "tool_calls", None):
                for tc in m.tool_calls:
                    args = tc.get("args", {}) or {}
                    keep = {k: args[k] for k in ("query","max_results","country","topic","search_depth") if k in args}
                    print(f"[Tool] {tc.get('name','tool')} {_trunc(keep)}")
                if i + 1 < len(msgs) and isinstance(msgs[i + 1], ToolMessage):
                    tm: ToolMessage = msgs[i + 1]
                    print(f"\n[Tool<-] {_trunc(tm.content)}")
                    i += 2
                    continue
            i += 1
    return response

# メインループ
async def main():
    llm = create_llm(model_id, base_url)

    thread_id = new_thread_id(os.getenv("USER_ID", "guest"))
    config = {"configurable": {"thread_id": thread_id}}

    async with AsyncExitStack() as stack:
        sessions: List[tuple[str, ClientSession]] = []
        for script in MCP_SCRIPTS:
            label = os.path.splitext(os.path.basename(script))[0]
            # Stdio指定
            server = StdioServerParameters(
                command=MCP_COMMAND,
                args=[script],
                env=_forward_env(),
            )
            # MCPサーバの起動
            stdio_rx, stdio_tx = await stack.enter_async_context(stdio_client(server))
            # セッション初期化
            session = await stack.enter_async_context(ClientSession(stdio_rx, stdio_tx))
            await session.initialize()
            sessions.append((label, session))
            print(f"[mcp] started: {label} ({script})")

        #MCPツール一覧をリスト化
        tools: List[Any] = []
        for label, session in sessions:
            t = await build_mcp_tools(session, label=label, namespace=MCP_NAMESPACE)
            tools.extend(t)

        print(f"[tools] {len(tools)} tools loaded:")
        for t in tools:
            print("  -", t.name)

        # エージェント実行途中の履歴のメモリ
        checkpointer = InMemorySaver()

        system_prompt = (
            "あなたは日本語で回答する汎用AIアシスタントです。"
            "ユーザーの意図を正確に把握し、根拠のある回答を簡潔に示してください。"
            "必要に応じて公開ツールを活用して情報を取得・加工します。"
            "ReAct（計画→実行→要約）の流れを保ち、推測は避け、出典や結果をわかりやすく提示してください。"
        )

        # ReActエージェント
        agent = create_react_agent(
            model=llm,
            tools=tools,
            state_schema=State,
            checkpointer=checkpointer,
            prompt=system_prompt,
        )

        print(f"\nMCP Agent Thread: {thread_id}")
        print("質問をしてください。終了するには 'exit' を入力。")

        while True:
            try:
                user_input = input("\n> ").strip()
            except (KeyboardInterrupt, EOFError):
                print("\nセッションを終了します。")
                break

            if not user_input:
                continue
            if user_input.lower() in {"exit", "quit", "q"}:
                print("セッションを終了します。")
                break

            resp = await run_agent(agent, user_input, config)
            if resp:
                final = resp["messages"][-1]
                print("\n[応答]\n" + getattr(final, "content", str(final)))

if __name__ == "__main__":
    asyncio.run(main())