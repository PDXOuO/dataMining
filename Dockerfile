FROM python:3.11-slim

LABEL description="图书馆个性化图书推荐 - 初赛提交模型包"

WORKDIR /app

# 依赖层单独缓存，源码变更不触发重装
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY scripts/ ./scripts/
COPY run.py ./
COPY data/ ./data/

# 生成提交文件并执行格式校验
CMD ["python", "run.py"]
