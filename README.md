# EBee的个人网站

一个记录 AI Agent 使用经历的个人网站。使用 HTML、CSS 和少量原生 JavaScript，无需安装依赖或构建。

## 页面

- `index.html`：首页、最新手记、待写选题。
- `notes.html`：文章列表。
- `about.html`：个人介绍。
- `posts/starting.html`：第一篇实践回顾及后续文章的版式参考。
- `assets/`：样式、移动端菜单、网站图标。

首篇文章为《我怎样用 Codex：从四个项目开始》。研究、知识库与工作流内容根据本人授权查阅的项目会话整理；Spotify 部分仍待核对具体经历。不要把待补充内容写成已完成成果。

## 本地查看

在这个目录运行：

```powershell
python -m http.server 8000 --bind 127.0.0.1
```

打开 <http://127.0.0.1:8000>。按 Ctrl+C 停止服务器。

## 更新文章

1. 复制 `posts/starting.html`，改成新的英文文件名，例如 `first-agent-task.html`。
2. 修改 `<title>`、描述、日期、文章标题和 `<article>` 正文。保留页面的相对资源路径。
3. 在 `notes.html` 中复制一条 `.archive-entry`，填写新文章链接、日期、标题、简介，并更新文章总数。
4. 在 `index.html` 中更新 `.featured-note` 的链接及内容，指向最新文章。
5. 检查文字、链接及个人信息后提交到网站仓库。GitHub Pages 从 main 分支根目录发布时会自动更新。

文章建议写清：任务背景、与 Agent 的协作过程、验证方式、实际结果、遇到的问题、下一次的改进。区分已完成、进行中和未验证的内容。

## GitHub Pages

仓库建议命名为 `agent-notes`。在仓库 **Settings → Pages** 中选择 **Deploy from a branch → main → /(root)**。保留 `.nojekyll` 文件。

如果使用其他仓库名或账号，需同步调整 `404.html` 的绝对首页地址以及全站 GitHub 个人页链接。

官方说明：https://docs.github.com/en/pages/getting-started-with-github-pages/creating-a-github-pages-site

## 线上地址

- 网站：https://michael-ebee.github.io/agent-notes/
- 源码：https://github.com/Michael-EBee/agent-notes
