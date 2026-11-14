# 发版流程（Release SOP）

govio 的版本发布 = **git tag + GitHub Release（四资产）+ PyPI**。本文件是 v0.7.2 发版过程固化下来的清单，照做即可，末尾附 v0.7.2 的实测记录作为参照。

> PyPI 发布由 **owner 手工执行**（`uv publish`），agent 不代发、本机也不存发布凭据。

## 1. 前置检查

```bash
git switch main && git pull --ff-only
gh run list --branch main --limit 3          # CI 必须全绿
uv lock --check                              # lock 与 pyproject 一致
uv run ruff check .                          # All checks passed!
uv run ruff format --check .                 # 0 files would be reformatted
uv run python -m pytest tests/ -q            # 记录基线（v0.7.2 为 482 passed / 18 skipped）
```

版本号一致性（**逐项人工核对**，打包脚本只注入产物、不改源清单与文档）：

| 位置 | 字段 |
|---|---|
| `pyproject.toml` | `project.version` |
| `docs/specs/README.md` | `- **Version**: x.y.z` |
| `packaging/pi/package.json` | `version` |
| `packaging/deepseek/package.json` | `version` |
| `docs/install.md` | `-V` 示例输出、Releases 下载链接、`dist/govio-pi-*`、`dist/govio-dsh-*`、`pi install`/`dsh plugin add` 的版本、`govio-cli ≥` 最低要求 |
| `packaging/pi/README.md`、`packaging/deepseek/README.md` | 同上（安装命令、产物路径、`govio-cli ≥`） |

> `scripts/_packaging_common.py`、`scripts/package_harness_plugin.py` 里「如 0.7.0」是**版本号格式示例**，不要跟着改。

## 2. 打 tag

```bash
git tag v0.7.2 <main-sha>     # 轻量 tag，与既有 v0.7.x 一致（git cat-file -t v0.7.1 → commit）
git push origin v0.7.2
```

## 3. 构建资产（干净树）

```bash
rm -f dist/govio-*.whl dist/govio-*.tar.gz dist/govio-skills.zip
uv build
uv run package_skills.py       # → dist/govio-skills.zip
```

**校验 `govio-skills.zip` 条目数 = 22**，且：

- 含 `skills/govio-observe/assets/visualize-relations-schema.json`（skill 自带资源，`.gitignore` 已加例外放行）
- 不含 `skills/govio/assets/**`（派生数据，由 `govio-cli meta graph` 生成，永不入包）

要严格验证就在**干净 clone** 里重打包并逐条目比 CRC：

```bash
git clone --branch main <repo> /tmp/rel-verify && cd /tmp/rel-verify
python package_skills.py
unzip -v dist/govio-skills.zip | awk 'NR>3 && $8!="" {print $7, $8}' | sort > /tmp/crc_clean.txt
# 与主工作树的同款输出 diff，应完全一致（zip 容器时间戳不同属正常）
```

> 历史坑：干净检出曾只有 **21** 条目 —— `.gitignore` 的宽泛 `assets/` 规则吞掉了上面那个 schema 文件，只有存在本地残留的开发机才打出 22 条目。分发类结论必须在干净检出里验证。

## 4. 四个 Release 资产（缺一不可）

| 资产 | 来源 |
|---|---|
| `govio-x.y.z-py3-none-any.whl` | `uv build` |
| `govio-x.y.z.tar.gz` | `uv build` |
| `govio-skills.zip` | `uv run package_skills.py` |
| `chocolate.zip` | **仓库 `data/chocolate.zip`** —— 被 gitignore、无构建脚本，只能人工带入 Release |

> v0.7.1 的 Release 只挂了 `govio-skills.zip`、缺 `chocolate.zip`，而 `docs/install.md` 的下载链接同时指向两者 → 该链接 404。**发版清单固定为这四个。**

## 5. 发布 GitHub Release

```bash
gh release create v0.7.2 \
  --title "govio 0.7.2 — <一句话摘要>" \
  --notes-file <notes.md> --latest \
  dist/govio-skills.zip dist/govio-0.7.2-py3-none-any.whl dist/govio-0.7.2.tar.gz data/chocolate.zip
```

发布说明建议包含：主线变更（按功能/重构/修复分组）、兼容性声明（CLI 子命令与参数、退出码、输出 JSON 结构、node_id 规则、CSV 格式、assets 契约是否变化）、测试基线、安装步骤、**资产 SHA256 表**、`compare/v0.7.1...v0.7.2` 链接。

## 6. 发布 PyPI（owner 手工）

```bash
uv publish dist/govio-0.7.2*
```

未发布前 `docs/install.md` 的 `uv tool install govio` 仍会装到旧版本 —— 在 Release 说明里注明可改用附带 wheel：

```bash
uv tool install --from govio-0.7.2-py3-none-any.whl govio --compile-bytecode -p 3.13 --force
```

## 7. 发布后验证

```bash
# 资产齐、状态正确
gh release view v0.7.2 --json tagName,isDraft,isPrerelease,assets \
  --jq '{tagName,isDraft,isPrerelease,assets:[.assets[]|{name,size}]}'

# install.md 引用的下载链接实测可用（不要只看状态码，要比哈希）
curl -sIL -o /dev/null -w "%{http_code}\n" \
  https://github.com/chenxinma/govio/releases/download/v0.7.2/govio-skills.zip
curl -sL -C - -o /tmp/choc.zip \
  https://github.com/chenxinma/govio/releases/download/v0.7.2/chocolate.zip
sha256sum /tmp/choc.zip        # 与本地 data/chocolate.zip 比对

# PyPI 版本
curl -s https://pypi.org/pypi/govio/json | jq -r .info.version

# 本地部署刷新
./start.sh && govio-cli -V
```

> 大资产（`chocolate.zip` 5.8MB）从 Release CDN 下载在弱网下可能被截断（曾实测只下到 4,853,004/5,802,728 字节，哈希自然不符），必须 `curl -C -` 续传后重算，别把网络中断误判成资产损坏。

## v0.7.2 实测记录（参照）

- tag `v0.7.2` @ `7587048`（轻量 tag）；Release 为 **Latest**，`isDraft=false`、`isPrerelease=false`、`targetCommitish=main`
- 四资产 SHA256：`govio-skills.zip` `2648d574…`（61,621 B / 22 条目）、`chocolate.zip` `9b965c35…`（5,802,728 B / 3 文件）、`govio-0.7.2-py3-none-any.whl` `579ac480…`（137,158 B）、`govio-0.7.2.tar.gz` `a7d005ca…`（4,212,086 B）
- 下载回来的 `govio-skills.zip` 与 `chocolate.zip` 哈希均与本地构建一致；`unzip -t chocolate.zip` 无错
- 构建可复现：`./start.sh` 重跑后 whl / tar.gz / skills.zip 三个哈希与已上传资产**逐一相同**
- 范围 `v0.7.1..v0.7.2`：caliber 增强（sqlparse 分词、has_star 修正、measure-first 降噪）、Phase 3 七项精简、CI 门禁、全仓格式化、datacompy/ruff 版本锁、skills 自带资源修复、版本号统一
