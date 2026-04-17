# PsAgent Frontend

## 启动

```bash
cd frontend
npm install
npm run dev
```

## 环境变量

复制：

```bash
cp .env.example .env
```

然后按需修改：

```text
VITE_API_BASE_URL=http://localhost:8000
```

## 当前预置内容

1. Vue 3 + Vite + TypeScript
2. 最小页面壳
3. 后端 API client
4. `POST /edit/stream` 的 SSE 解析骨架
5. 当前后端接口类型定义

## 推荐下一步

1. 做上传区
2. 做任务进度时间线
3. 做结果图对比
4. 做人工审核弹窗
