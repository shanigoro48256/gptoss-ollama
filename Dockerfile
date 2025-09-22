FROM nvidia/cuda:12.8.0-runtime-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive TZ=Etc/UTC \
    UV_NO_SPINNER=1 UV_LINK_MODE=copy \
    LANG=C.UTF-8 LC_ALL=C.UTF-8 PYTHONUTF8=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    git nano vim lsof curl ca-certificates tini python3.12 python3.12-venv python3-pip \
 && rm -rf /var/lib/apt/lists/*

RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:${PATH}"
RUN ln -sf /root/.local/bin/uv /usr/local/bin/uv

WORKDIR /app

COPY pyproject.toml ./

RUN mkdir -p /app/src && printf "" > /app/src/__init__.py

RUN uv python install 3.12 \
 && uv venv --python 3.12 --seed /app/venv \
 && /app/venv/bin/python -m pip install --upgrade pip wheel setuptools \
 && uv pip install --python /app/venv/bin/python -U \
      openai>=1.99.9 openai-harmony>=0.0.4 openai-agents>=0.2.6 \
      langgraph>=0.6.5 langgraph-prebuilt>=0.6.4 langgraph-checkpoint>=2.1.1 \
      langchain>=0.3.27 langchain-ollama>=0.3.6 \
      mcp>=1.12.4 fastmcp>=2.11.3 \
      tavily-python>=0.7.10 \
      python-dotenv>=1.0.1 pydantic>=2.11.7 httpx>=0.28.1 \
 && uv pip install --python /app/venv/bin/python -U \
      jupyter jupyterlab "nbformat>=5.10,<6" ipykernel \
 && test -x /app/venv/bin/jupyter \
 && /app/venv/bin/python -c "import jupyterlab; print('JupyterLab OK', jupyterlab.__version__)"

ENV VIRTUAL_ENV=/app/venv
ENV PATH="/app/venv/bin:/root/.local/bin:${PATH}"

CMD ["/bin/bash"]

