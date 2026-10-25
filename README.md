# EBee的个人网站

一个记录 AI Agent 使用经历的个人网站。使用 HTML、CSS 和少量原生 JavaScript，无需安装依赖或构建。

## 页面

- `index.html`：首页、最新手记、五项实践入口。
- `notes.html`：文章列表。
- `about.html`：个人介绍。
- `posts/starting.html`：五个项目的总览与详细文章入口。
- `posts/spotify-design.html`、`song-research.html`、`knowledge-base.html`、`engineering-workflow.html`、`game-mod.html`：五篇项目手记。
- `assets/`：样式、移动端菜单、网站图标。

现有六篇手记：一篇总览和五篇项目回顾。内容依据本人授权查阅的项目会话与相关材料整理。

- Spotify：依据“简历修改”项目的完整相关讨论及历史简历，区分榜单数据清洗、五项指标与图表产物的历史分析介绍，以及垂类热榜、评分卡和对话式选曲工具的后续方案，不把推演功能当作已完成部署。
- 歌曲研究：保留固定样本、未确认记录和来源缺口的边界。
- 个人知识库：检索数据仅描述当时固定问题集的验收结果，知识资产化框架不等于真实产品已完成交付。
- 工程工作流：描述历史规则与简化过程，不替代各项目当前仓库的规则。
- 游戏 Mod：基于已归档的 More Enemies 会话。ARK 修复有当时用户实测反馈，后续 Camp2／桥面候选仍未完成同等实机验证；不分发游戏或 Mod 资源。

## 本地查看

在这个目录运行：

```powershell
python -m http.server 8000 --bind 127.0.0.1
```

打开 <http://127.0.0.1:8000>。按 Ctrl+C 停止服务器。

## 更新文章

1. 复制一篇 `posts/` 中的项目手记（例如 `spotify-design.html`），改成新的英文文件名，例如 `first-agent-task.html`。
2. 修改 `<title>`、描述、日期、文章标题和 `<article>` 正文。保留页面的相对资源路径。
3. 在 `notes.html` 中复制一条 `.archive-entry`，填写新文章链接、日期、标题、简介，并更新文章总数。
4. 在 `index.html` 中更新 `.featured-note` 的链接及内容，指向最新文章；项目入口变化时同步更新首页和关于页。
5. 检查文字、链接及个人信息后提交到网站仓库。GitHub Pages 从 main 分支根目录发布时会自动更新。

文章建议写清：任务背景、与 Agent 的协作过程、验证方式、实际结果、遇到的问题、下一次的改进。区分已完成、进行中和未验证的内容。

## GitHub Pages

仓库建议命名为 `agent-notes`。在仓库 **Settings → Pages** 中选择 **Deploy from a branch → main → /(root)**。保留 `.nojekyll` 文件。

如果使用其他仓库名或账号，需同步调整 `404.html` 的绝对首页地址以及全站 GitHub 个人页链接。

官方说明：https://docs.github.com/en/pages/getting-started-with-github-pages/creating-a-github-pages-site

## 线上地址

- 网站：https://michael-ebee.github.io/agent-notes/
- 源码：https://github.com/Michael-EBee/agent-notes
