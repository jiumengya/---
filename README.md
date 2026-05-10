<div align="center">

<h1>语伴 - 虚拟伴侣</h1>

你的 AI 虚拟伴侣 · 可自定义人设 · 本地运行 · 隐私安全

<img src="https://img.shields.io/badge/Python-3.10+-blue?style=for-the-badge&logo=python&logoColor=white" alt="Python">
<img src="https://img.shields.io/badge/FastAPI-0.100+-009688?style=for-the-badge&logo=fastapi&logoColor=white" alt="FastAPI">
<img src="https://img.shields.io/badge/License-AirExchange-orange?style=for-the-badge" alt="License">
<img src="https://img.shields.io/badge/Platform-Windows-0078D6?style=for-the-badge&logo=windows&logoColor=white" alt="Windows">

</div>

---

## 这是什么

**语伴** 是一个跑在你电脑上的 AI 虚拟伴侣。

她有自己的名字、性格和小窝，会陪你聊天、发动态、记住你说过的话。
**所有数据都在你本地，没有服务器，没有隐私问题。**

> 接入你自己的 API Key，想多甜就多甜。

---

## 能做什么

<table>
<tr>
  <td width="50%">

### 聊天
文字 + 图片，`Enter` 发送，`Shift+Enter` 换行。
聊天记录自动保存。

### 小窝
AI 会自动发日常动态，每天 1-2 条。
你可以点赞、评论，像朋友圈一样。

  </td>
  <td width="50%">

### 自定义人设
给 AI 起名字，定性格，设称呼。
温柔？傲娇？治愈？你说了算。

### 气泡定制
自己挑形状和颜色，
AI 和你的气泡可以不一样。

  </td>
</tr>
<tr>
  <td width="50%">

### 记忆系统
AI 会从对话里提取重要信息，
下次聊天她还会记得。

### 技能系统
可扩展的工具调用能力，
让 AI 不止会聊天。

  </td>
  <td width="50%">

### 图片生成
动态可以自动配图，
配好图片模型就能用。

### 微信接入
扫码绑微信，
手机上也能和她聊。

  </td>
</tr>
</table>

---

## 一条命令跑起来

```bash
# 进入目录
cd yu-plat

# 激活虚拟环境
.\venv\Scripts\activate

# 启动
yu app
```

浏览器打开 **http://localhost:8000**

不想开终端？直接双击 `启动.bat`。

| 命令 | 干嘛的 |
|------|--------|
| `yu app` | 启动 |
| `yu stop` | 停止 |
| `yu status` | 看状态 |

---

## 三步上手

**1. 配模型** - 设置页填 API Key，支持 阿里云百炼 / DeepSeek / 任何 OpenAI 兼容接口

**2. 设人设** - 起名字、定称呼、选性格

**3. 开始聊** - 回到聊天页，Enter 发送

---

## 项目结构

```
yu-plat/
├── yu_body/
│   ├── server.py          # FastAPI 主服务
│   ├── user_memory.py     # 记忆系统
│   └── yu_skills.py       # 技能系统
├── index.html             # 前端（纯 HTML/CSS/JS，无框架）
├── setup.py               # 包配置
├── cli.py                 # 命令行入口
├── venv/                  # 虚拟环境（开箱即用）
├── 启动.bat                # 双击启动
└── .gitignore
```

---

## 依赖

```
fastapi >= 0.100
uvicorn >= 0.20
httpx >= 0.24
pydantic >= 2.0
Pillow >= 9.0
```

---

## FAQ

**Q: 聊天没反应？**
去设置页点「测试连接」，确认 API Key 和地址正确。

**Q: 怎么重置？**
删掉 `yu_history.json`、`yu_moments.json`、`yu_profile.json`，重启。

**Q: 支持 macOS / Linux 吗？**
理论上只要装好 Python 依赖就能跑，`启动.bat` 是 Windows 的。

---

## 致谢

本项目微信连接功能使用了 [qwenpaw](https://github.com/HansHans135/qwenpaw) 的 ClawBot 部分代码，感谢开源社区。

---

## License

AirExchange
