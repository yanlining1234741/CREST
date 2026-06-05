# 上传到 GitHub 教程 — CREST

本地仓库路径：`/sda/yxyang/qsba_workspace/CREST-github`

论文全称：**CREST: Collapse-Free Routing with Equipartitioned Semantic Buckets for Cross-Modal Retrieval**

## 第一步：在 GitHub 创建空仓库

1. https://github.com/new
2. **Repository name**: `CREST`（或 `CREST-CVPR25`）
3. **Public**，不要添加 README / .gitignore
4. 记下地址：`https://github.com/你的用户名/CREST.git`

## 第二步：本地提交

```bash
cd /sda/yxyang/qsba_workspace/CREST-github

git config user.name "yanlining1234741"
git config user.email "1234741@wku.edu.cn"

export CREST_DATA_ROOT=/sda/yxyang/qsba_workspace/mbeir_aligned
python scripts/setup_mbeir_configs.py

git add -A
git status    # 确认无 *.pt 大文件
git commit -m "CREST: official release with reproduction guide and results"
```

## 第三步：推送

```bash
git branch -M main
git remote add origin https://github.com/yanlining1234741/CREST.git
git push -u origin main
```

HTTPS 推送时 Password 填 **Personal Access Token**（`repo` 权限）。

SSH：

```bash
git remote set-url origin git@github.com:yanlining1234741/CREST.git
git push -u origin main
```

## 后续更新

```bash
git add -u && git commit -m "your message" && git push
```
