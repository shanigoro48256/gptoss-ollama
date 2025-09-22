import os
from pathlib import Path
from typing import Dict
from fastmcp import FastMCP

# セーフガード
BASE_DIR = Path(os.getenv("AGENT_WORKDIR", ".")).resolve()
ALLOWED_WRITE_EXTS = {".txt", ".md", ".json", ".yaml", ".yml"}
MAX_WRITE_BYTES = 256 * 1024  # 256KB

def _safe_path(p: str) -> Path:
    path = (BASE_DIR / p).resolve()
    if not str(path).startswith(str(BASE_DIR)):
        raise ValueError(f"安全のため '{p}' へのアクセスを拒否しました")
    return path

# FastMCPのインスタンス化
mcp = FastMCP(name="fs mcp server")

# MCPツール
@mcp.tool(
    name="read_base_txt",
    tags=["fs", "read", "text", "txt", "nosubdir"],
    description="BASE_DIR 直下の *.txt を最大 max_files 件まで読み取り、{ファイル名: 内容} を返す。サブディレクトリは対象外。"
)
def load_texts_top(
    pattern: str = "*.txt",
    max_files: int = 20,
    max_bytes_per_file: int = 80_000,
    encoding: str = "utf-8",
) -> Dict[str, str]:
    """
    BASE_DIR 直下の pattern に合うファイルを最大max_files 件読み込み、
    {ファイル名: 内容(最大 max_bytes_per_file)} を返す。サブディレクトリは対象外。
    """
    paths = [p for p in sorted(BASE_DIR.glob(pattern)) if p.is_file()][:max_files]
    out: Dict[str, str] = {}
    for p in paths:
        data = p.read_bytes()[:max_bytes_per_file]
        out[p.name] = data.decode(encoding, errors="replace")
    return out

@mcp.tool(
    name="write_text_file",
    tags=["fs", "write", "text", "txt", "md", "json", "yaml", "overwrite"],
    description="テキストを書き込む（拡張子: .txt/.md/.json/.yaml/.yml、最大256KB）。既存の場合は overwrite=True が必要。"
)
def fs_write(path: str, content: str, overwrite: bool = False) -> str:
    """
    テキストを書き込む（拡張子: .txt/.md/.json/.yaml/.yml、最大256KB）。
    """
    fp = _safe_path(path)
    if fp.suffix.lower() not in ALLOWED_WRITE_EXTS:
        return f"許可されていない拡張子です: {fp.suffix}（許可: {sorted(ALLOWED_WRITE_EXTS)}）"
    b = content.encode("utf-8")
    if len(b) > MAX_WRITE_BYTES:
        return f"サイズ上限超過: {len(b)} bytes > {MAX_WRITE_BYTES} bytes"
    if fp.exists() and not overwrite:
        return "既に存在します（overwrite=True で上書き可）"
    fp.parent.mkdir(parents=True, exist_ok=True)
    fp.write_bytes(b)
    return f"書き込み完了: {fp.relative_to(BASE_DIR)} / {len(b)} bytes"

# エントリポイント
if __name__ == "__main__":
    print("[search] Fs MCP server starting (stdio)...")
    mcp.run()
