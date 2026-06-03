# 上传到 GitHub 教程

本地仓库路径：`/sda/yxyang/qsba_workspace/QSBA-github`

## 第一步：在 GitHub 网站创建空仓库

1. 登录 https://github.com
2. 右上角 **+** → **New repository**
3. 填写：
   - **Repository name**: `QSBA`（或 `QSBA-CVPR25`）
   - **Public**
   - **不要**勾选 “Add a README” / “Add .gitignore”（本地已有）
4. 点 **Create repository**
5. 记下页面上的地址，例如：`https://github.com/你的用户名/QSBA.git`

## 第二步：本地初始化并提交

在服务器终端执行：

```bash
cd /sda/yxyang/qsba_workspace/QSBA-github

# 生成可移植的 config（可选，提交前把绝对路径改成你的 data 路径）
export QSBA_DATA_ROOT=/sda/yxyang/qsba_workspace/mbeir_aligned
python scripts/setup_mbeir_configs.py

git init
git add .
git status    # 确认没有 *.pt 大文件（已被 .gitignore 排除）

git commit -m "Initial release: QSBA code, reproduction guide, and paper results"
```

若 `git status` 里出现几百 MB 的 `.pt` 文件，**不要提交**，检查 `.gitignore`。

## 第三步：关联远程并推送

```bash
# 把 YOUR_USER 和 REPO 换成你的
git branch -M main
git remote add origin https://github.com/YOUR_USER/QSBA.git

# 首次推送（会提示登录）
git push -u origin main
```

### 认证方式（二选一）

**A. HTTPS + Personal Access Token（推荐）**

1. GitHub → Settings → Developer settings → Personal access tokens → Generate (classic)
2. 勾选 `repo` 权限
3. 推送时：
   - Username：你的 GitHub 用户名
   - Password：粘贴 **token**（不是登录密码）

**B. SSH**

```bash
ssh-keygen -t ed25519 -C "your_email@example.com"
cat ~/.ssh/id_ed25519.pub   # 复制到 GitHub → Settings → SSH keys

git remote set-url origin git@github.com:YOUR_USER/QSBA.git
git push -u origin main
```

## 第四步：之后更新代码

```bash
cd /sda/yxyang/qsba_workspace/QSBA-github
# 修改代码后
git add -u
git commit -m "Describe your change"
git push
```

## 可选：发布 model checkpoint（大文件）

`.pt` 权重默认 **不进 git**。若要上传：

```bash
# 安装 Git LFS（一次性）
git lfs install
git lfs track "*.pt"
git add .gitattributes
git add path/to/router_K512.pt
git commit -m "Add VN router checkpoint via LFS"
git push
```

或把 checkpoint 放到 Zenodo / Hugging Face，在 README 里放下载链接（更常见）。

## 仓库大小检查

```bash
du -sh .
git count-objects -vH
```

当前代码 + JSON + 图约 **1–2 MB**，适合直接 push；数据需用户自行下载。
