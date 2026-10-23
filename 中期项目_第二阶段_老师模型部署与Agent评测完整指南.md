# A3 期中项目：老师发布的 SmolVLA / π0.5 部署、多回合评测与 Agent 实验指南

发布资源核对日期：2026-10-21；指南补充与脚本检查：2026-10-23。本文固定已核对的资源版本，不自动追随新上传版本。适用对象：已完成基础仿真和 expert 检查，准备运行老师新发布模型的你。

本文与《中期项目_第二阶段详细指南与论文调研.md》配套：ACT 继续按原文档做；这里专门处理新发布的 SmolVLA 和 π0.5。

## 0. 先看这一页：你到底要完成什么

本阶段的目标是：**下载任务权重 → 确认正确加载 → 单回合试运行 → 多场景评测 → 保存失败证据 → Agent 提出一个修改 → 用相同场景验证 → 留出场景最终验收。**

你说的“老师给的基座模型”，需要区分两类：

| 资源 | 实际含义 | 本阶段是否必需 |
|---|---|---|
| `Eter0109/a3-cookie-multi-left8-40k-v1` | A3 任务训练后的 SmolVLA 权重 | 是，先部署它 |
| `yunsx114/a3-dual-arm-cookie-pi05-aligned` | A3 任务微调后的 π0.5 权重，在 ModelScope | 是，作为第二条部署路线 |
| `lerobot/smolvla_base` / `lerobot/pi05_base` | 通用基座 | 本阶段不需要重新下载整个基座来训练 |
| `Eter0109/a3-front-close-left-100` | 100 条示范数据 | ACT 训练用；不是两个发布模型推理时必须另下的数据 |

发布权重已携带归一化统计量。**不要拿 ACT 那份 100 条数据的统计量去替换老师任务权重自己的统计量。** 两者训练数据不相同。

### 0.1 先做哪一个

1. 在你的 WSL 笔记本上先做 **SmolVLA**。
2. π0.5 用更大显存的 Linux GPU 机器。此次再次读取到你的 GPU 是 RTX 5060 Laptop、8151 MiB 显存。π0.5 权重文件本身约 9.35 GB，普通整模型 GPU 部署装不进这张 8 GB 卡。
3. π0.5 建议使用 24 GB 或以上显存起步，32 GB 更宽裕；这是资源规划建议，不是实测显存承诺。还要留出激活、渲染和加载内存，主机内存建议 32–64 GB。不要仅按权重文件大小租一台刚好够的机器。
4. 不建议把量化、CPU offload、从头训练都塞进首次部署；这些会额外改变实验条件。本指南未实现它们。

### 0.2 我已经检查了什么，哪些还没有做

- 检查了发布模型配置、processor、π0.5 模型卡、公开 `lerobot==0.5.1` 源码及本地评测接口。
- GitHub 查询到的上游 HEAD 为 `08ebe8cb370856051d0b32b07e691cc9b056716b`；本地外层项目 Git HEAD 为 `569a2639d62c8937291b5141c1c6d21a096aaac8`。本地 Git 根目录是 `D:/School/Agent/project`，不是它下面的源码子目录，因此不要把两个提交号当成同一条仓库历史。
- 当前公开评测 CLI 有 `smolvla:`、`act:` 和 `module:factory`；没有现成 `pi05:` 命令。
- 本指南附了独立脚本，不修改老师仓库的评分器、场景或 ACT 代码。
- **没有替你下载多 GB 权重，没有执行这两个模型的真实推理，也没有测出它们的成功率。** 文中命令是待你按关卡执行的流程；配套脚本的检查情况见同目录 `老师模型部署配套/验证记录.md`。

### 0.3 配套文件

它们已经放在 `D:\School\Agent\project\老师模型部署配套`：

| 文件 | 功能 |
|---|---|
| `prepare_checkpoint.py` | 在新目录创建部署副本，适配路径和已核对的配置字段 |
| `download_pi05.py` | 下载固定版本的 π0.5，每个文件检查大小及 SHA256 |
| `teacher_policy.py` | 两个策略的显式输入适配和严格权重加载 |
| `evaluate_teacher.py` | 使用仓库原有 2×5 场景和判定器进行统一评测 |
| `summarize_results.py` | 从真实结果生成逐回合 CSV 和简短 JSON |
| `compare_runs.py` | 核对固定条件后，按 seed 比较基线与 horizon 候选 |
| `research/` | 本次兼容性核查使用的源码和小配置；不是模型权重，不需要运行 |

**本指南出现的 `evaluate_teacher.py` 等是本次新增配套工具，不是老师仓库自带入口。**

## 1. 路线和完成标准

| 阶段 | 你做什么 | 通过标准 |
|---|---|---|
| 环境 | 安装独立环境、验证 CUDA | `pip check` 通过，实际 GPU 矩阵运算成功 |
| 资源 | 下载固定版本、创建部署副本 | 原始文件保留，生成部署清单，无未知字段被忽略 |
| 加载 | 严格匹配模型参数 | 日志出现 `STRICT_WEIGHTS_LOADED`，无权重缺失错误 |
| 接口冒烟 | 1 回合、20 控制步 | 有完整 `result.json`、有限动作、无异常 |
| 单回合 | 1 回合、500 步 | 能正常结束，查看最终分数；不要求一定成功 |
| 开发评测 | 固定 5 个场景 | 5 行结果齐全，不漏失败回合 |
| Agent 实验 | 分析证据、提出一个修改并验证 | 保存输入、理由、实际命令、结果和决策 |
| 确认/验收 | 20 个确认场景，再用留出场景 | 报告任务成功率与搬入数量，不把 5 回合当稳定结论 |

20 步只是检查接口；500 步是控制预算，在 20 Hz 下对应 25 秒仿真时间，不代表电脑上只用 25 秒。

## 2. SmolVLA：准备 WSL 终端和独立环境

以下命令都在 **WSL Ubuntu 的 Bash** 中执行，不要直接粘贴进 Windows PowerShell。

操作方法：打开 Windows Terminal，在标签页下拉菜单选择 Ubuntu；若现在处于 PowerShell，先输入 `wsl`，看到 Linux 的 shell 提示符后再执行下面命令。一次复制一个代码块，等命令结束并检查输出后，再执行下一块。代码行尾的 `\` 是续行符，后面不要手动加空格。

### 2.1 设置目录，每次新开终端都需要

```bash
cd /mnt/d/School/Agent/project/a3_dual_arm_sim
export PROJECT="$PWD"
export TOOLS="$(realpath ../老师模型部署配套)"
export RUN="$PROJECT/outputs/teacher_models_01"
export SMOL_RAW="$PROJECT/models/teacher_smol_raw"
export SMOL="$PROJECT/models/teacher_smol_deploy"
export SMOL_DEP="$PROJECT/models/dependencies/smolvlm_config"
export MUJOCO_GL=egl
set -o pipefail
mkdir -p "$RUN/logs"
printf 'PROJECT=%s\nTOOLS=%s\nRUN=%s\n' "$PROJECT" "$TOOLS" "$RUN"
test -f "$TOOLS/evaluate_teacher.py"
```

首次使用 `teacher_models_01`，新的整组实验可换 `teacher_models_02`。脚本拒绝覆盖已有单次实验目录；不要为重跑而删除旧结果。

### 2.2 安装环境

你之前 ACT 环境中的 Torch 已能导入，当前版本为 `2.10.0+cu128`。这里仍创建单独环境，避免为部署 VLA 改乱正在使用的 ACT 环境。pip 缓存通常能复用已下载的软件包。

首次安装执行以下整个代码块；之后回来继续实验，只需 `source ~/.venvs/a3-teacher-smol/bin/activate`，不必每次重新安装。若 `python3.12: command not found`，先执行 `python3 --version`；显示 3.12 时可将建环境那条命令的 `python3.12` 改为 `python3`。不是 3.12 时先准备该版本，别在另一个不确定版本的环境中继续。

```bash
python3.12 -m venv ~/.venvs/a3-teacher-smol
source ~/.venvs/a3-teacher-smol/bin/activate
python -m pip install -U pip -i https://pypi.tuna.tsinghua.edu.cn/simple
printf 'torch==2.10.0\ntorchvision==0.25.0\n' > "$RUN/torch_constraints.txt"

python -m pip install torch==2.10.0 torchvision==0.25.0 \
  --index-url https://download.pytorch.org/whl/cu128 \
  2>&1 | tee "$RUN/logs/install_torch.log"

python -m pip install -c "$RUN/torch_constraints.txt" \
  -i https://pypi.tuna.tsinghua.edu.cn/simple -e '.[train,dev]' \
  2>&1 | tee "$RUN/logs/install_smol.log"

python -m pip check
python -m pip freeze > "$RUN/environment_smol.txt"
```

一条安装命令报错就先解决，不要继续粘贴下一条。`pipefail` 使 `tee` 不掩盖安装失败，但交互终端不会自动阻止你手动执行下一条命令。

清华源用于普通 Python 依赖；它不会加速后面的 Hugging Face / ModelScope 模型下载。这里也没有修改全局 pip 源。

### 2.3 检查实际 GPU 计算

```bash
python - <<'PY'
import torch
from importlib.metadata import version
print('torch:', torch.__version__, 'lerobot:', version('lerobot'))
assert version('lerobot') == '0.5.1'
assert torch.cuda.is_available(), 'CUDA unavailable'
x = torch.randn(512, 512, device='cuda')
y = x @ x
torch.cuda.synchronize()
assert torch.isfinite(y).all()
print('CUDA compute OK:', torch.cuda.get_device_name())
PY
python "$TOOLS/evaluate_teacher.py" --help
```

## 3. SmolVLA：下载老师任务权重和辅助配置

### 3.1 下载固定版本的任务权重

```bash
hf download Eter0109/a3-cookie-multi-left8-40k-v1 \
  --revision ffa4dfbc3e36796a4b02bcf07e6776f6077e49b8 \
  --local-dir "$SMOL_RAW" \
  2>&1 | tee "$RUN/logs/download_smol.log"
```

这是任务权重，下载后目录中应包含 `model.safetensors`、`config.json`、两个 processor JSON、归一化状态及 `tokenizer/`。

目录名 `left8` 表示网络预测左臂 8 个量：7 个关节目标和 1 个夹爪量。环境需要双臂 16 维动作，因此配套脚本补上回合开始时的右臂保持姿态；没有加入专家几何规划。

### 3.2 只下载 VLM 配置与 processor/tokenizer 辅助文件

老师权重已经包含 VLM 参数，不需要再次下载整个 SmolVLM 权重。但构建模型还要读取 VLM 结构和图像处理配置。

```bash
python - <<'PY'
import os
from huggingface_hub import snapshot_download
p = snapshot_download(
    'HuggingFaceTB/SmolVLM2-500M-Video-Instruct',
    revision='7b375e1b73b11138ff12fe22c8f2822d8fe03467',
    local_dir=os.environ['SMOL_DEP'],
    allow_patterns=['*.json', '*.txt', '*.model', '*.jinja'],
)
print('Auxiliary config downloaded:', p)
PY
```

这里不下载 `.safetensors`。若下载网络失败，重试同一版本；不要换成另一个结构的 VLM 配置。

### 3.3 创建部署副本，不修改老师原始文件

```bash
python "$TOOLS/prepare_checkpoint.py" \
  --source "$SMOL_RAW" --dest "$SMOL" --dependency "$SMOL_DEP" \
  2>&1 | tee "$RUN/logs/prepare_smol.log"
```

脚本做了以下适配，并写入 `deployment_manifest.json`：

- 将发布者机器上的 VLM 绝对路径替换为你下载的辅助配置路径。
- 将 processor 中相对的 `tokenizer` 路径替换为部署副本中的绝对路径。
- 只对 `expert_variant=base` 的发布配置，移除公开 LeRobot 0.5.1 不认识的已核对扩展字段。不会随意删除未知字段，也不会修改网络层数、动作维度、归一化统计量或训练权重。
- 大权重用 Linux 符号链接指向原始目录，避免复制占空间。**原始目录不能删除；迁移机器后必须重新准备部署副本。**
- 真正的网络结构兼容还要靠下一步严格加载验证。配置能解析不等于权重已经正确加载。

如果报 `Unreviewed config keys`、`size mismatch`、`Missing key`：保存错误并停止本阶段。不要用 `strict=False` 绕过，不要把失败改成“加载成功”。这可能意味着老师的训练实现与公开 LeRobot 版本仍有差异，需要依据具体报错核对训练版本。

## 4. SmolVLA：从 20 步到一个完整回合

### 4.1 20 步接口冒烟

```bash
python "$TOOLS/evaluate_teacher.py" \
  --model smolvla --checkpoint "$SMOL" --device cuda \
  --horizon 50 --flow-steps 25 --inference-seed 1000 \
  --episodes 1 --seed-start 21999 --max-steps 20 \
  --out "$RUN/smol_smoke20" \
  2>&1 | tee "$RUN/logs/smol_smoke20.log"
```

通过时应看到 `STRICT_WEIGHTS_LOADED`，随后生成 `smol_smoke20/result.json`。这个结果很可能为 0 分或失败，20 步本来就不足以搬完。

如果只有 `run.json` 和 `error.json`，没有 `result.json`，说明没有完成评测，不应把它算成模型任务成功率里的 0 分回合。

执行后用这几条核对文件：

```bash
test -s "$RUN/smol_smoke20/result.json" && echo '完整报告已生成'
tail -n 40 "$RUN/logs/smol_smoke20.log"
python "$TOOLS/summarize_results.py" "$RUN/smol_smoke20/result.json"
```

通过标准是：正常完成 1 回合、实际执行了预期控制步或记录了明确的提前终止原因、没有加载/非有限动作异常。没有 `result.json` 时不要运行汇总，把 `error.json` 和日志末尾交给 Agent 排查。

本文基线沿用发布配置 `horizon=50`：每次预测 50 个动作后依次执行，再看新画面规划。后面可以研究改为 8 或 4；**先不要把它改掉然后称为“发布默认结果”。**

SmolVLA 的任务文本在配套脚本中采用仓库现有 2×5 简短指令 `transfer 10 cookies into target box`，并保存到报告。发布训练清单含 11 个任务标签，但本次未获取这 11 条原始文本；因此它是本文明确指定的起始指令，不声称已经逐字复现老师该 checkpoint 的训练指令。若模型加载正常却表现异常，把原始任务文本核对作为优先排查项。

### 4.2 一个完整预算回合，同时录制诊断

```bash
python "$TOOLS/evaluate_teacher.py" \
  --model smolvla --checkpoint "$SMOL" --device cuda \
  --horizon 50 --flow-steps 25 --inference-seed 1000 \
  --episodes 1 --seed-start 22000 --max-steps 500 --diagnostic \
  --out "$RUN/smol_demo500" \
  2>&1 | tee "$RUN/logs/smol_demo500.log"

python "$TOOLS/summarize_results.py" "$RUN/smol_demo500/result.json"
```

查看 `smol_demo500/trace/seed_22000/` 中的 `front.mp4`、`left_wrist.mp4`、`right_wrist.mp4`、`trace.jsonl` 和诊断 JSON。

观察顺序：机械臂有没有接近饼干 → 有没有抓住 → 是否带着饼干移动 → 是否放入目标盒 → 是否返回抓第二批。单凭最终 5 分不能断定“忘记第二批”，也可能是第二批掉落，需要视频确认。

## 5. π0.5：先选机器，再部署

### 5.1 机器和源码保持一致

在大显存 Linux 机器上部署。你可以使用已有的课程 GPU 资源；本指南不替你租机器，也不读取或登录 `account.txt` 中的账号。

为了公平比较，优先把**本次本地源码快照**传到服务器，而不是服务器另拉一个已经更新的 main。两个模型必须使用同一套场景、判定和配置。

在本地 WSL 打包（包含机器人 XML/mesh，排除运行产物和下载权重）：

```bash
cd "$PROJECT"
tar --exclude='./outputs' --exclude='./datasets' --exclude='./.git' \
  --exclude='./.venv' --exclude='./models/teacher_*' \
  --exclude='./models/dependencies' --exclude='__pycache__' \
  -czf "$RUN/a3_source_snapshot.tar.gz" .
sha256sum "$RUN/a3_source_snapshot.tar.gz" > "$RUN/a3_source_snapshot.sha256"
```

用课程服务器的文件上传功能，上传：

1. `a3_source_snapshot.tar.gz` 和对应 SHA256 文件；
2. 整个 `老师模型部署配套` 文件夹（`research/`、`validation/` 可以不传）。

服务器上建立目录，例如 `~/a3_teacher_project/a3_dual_arm_sim`，将源码压缩包解压到这个目录；把配套文件夹放在 `~/a3_teacher_project/老师模型部署配套`。不要把模型权重一并打包上传，服务器直接下载更合适。

具体操作：先在服务器执行下面两条，再通过上传界面将压缩包放入 `~/a3_teacher_project`。如果该目录已有另一套项目，换一个全新目录名，下面路径也同步修改。

```bash
mkdir -p ~/a3_teacher_project
mkdir ~/a3_teacher_project/a3_dual_arm_sim
```

上传后执行：

```bash
cd ~/a3_teacher_project
sha256sum a3_source_snapshot.tar.gz
tar -xzf a3_source_snapshot.tar.gz -C a3_dual_arm_sim
test -f a3_dual_arm_sim/pyproject.toml
test -f 老师模型部署配套/evaluate_teacher.py
```

将服务器 `sha256sum` 输出的前 64 个字符与本地 `.sha256` 文件中的值比较，应完全相同。本地校验文件带的是本地绝对路径，不能直接在服务器执行 `sha256sum -c` 而期待找到同一路径。两个 `test` 没输出且返回成功是正常的；如果提示文件不存在，先修正上传层级。

如果你选择服务器从 GitHub clone，则两个模型都应在该服务器同一个固定提交上重新跑，不能直接把两套不同源码下的分数比较。

### 5.2 在服务器新终端设置目录

以下路径按上面示例填写；若你选了其他目录，修改 `cd` 那一行。

```bash
cd ~/a3_teacher_project/a3_dual_arm_sim
export PROJECT="$PWD"
export TOOLS="$(realpath ../老师模型部署配套)"
export RUN="$PROJECT/outputs/teacher_models_01"
export PI_RAW="$PROJECT/models/teacher_pi05_raw"
export PI="$PROJECT/models/teacher_pi05_deploy"
export PI_DEP="$PROJECT/models/dependencies/paligemma_tokenizer"
export MUJOCO_GL=egl
set -o pipefail
mkdir -p "$RUN/logs"
nvidia-smi
df -h .
```

检查空闲显存，而不是只看显卡标称容量。准备约 20 GB 以上空闲磁盘作为模型和缓存起点，视频、数据和其他模型还需额外空间。

### 5.3 安装 π0.5 专用环境

```bash
python3.12 -m venv ~/.venvs/a3-teacher-pi05
source ~/.venvs/a3-teacher-pi05/bin/activate
python -m pip install -U pip -i https://pypi.tuna.tsinghua.edu.cn/simple
printf 'torch==2.10.0\ntorchvision==0.25.0\n' > "$RUN/torch_constraints.txt"
python -m pip install torch==2.10.0 torchvision==0.25.0 \
  --index-url https://download.pytorch.org/whl/cu128 \
  2>&1 | tee "$RUN/logs/install_pi_torch.log"
python -m pip install -c "$RUN/torch_constraints.txt" \
  -i https://pypi.tuna.tsinghua.edu.cn/simple \
  -e '.[dev]' 'lerobot[pi]==0.5.1' \
  2>&1 | tee "$RUN/logs/install_pi.log"
python -m pip check
python -m pip freeze > "$RUN/environment_pi05.txt"
python -c "import torch; from lerobot.policies.pi05.modeling_pi05 import PI05Policy; print(torch.__version__, torch.cuda.is_available())"
```

再执行第 2.3 节的 GPU 矩阵运算检查。服务器驱动若不支持这套 CUDA wheel，先解决驱动/运行时兼容；不要在不记录的情况下随意降级一大组依赖。

### 5.4 下载固定版本 π0.5

```bash
python "$TOOLS/download_pi05.py" --dest "$PI_RAW" \
  2>&1 | tee "$RUN/logs/download_pi05.log"
```

固定资源：ModelScope `yunsx114/a3-dual-arm-cookie-pi05-aligned`，revision `6b48a8f942dafde5e3add8e6c0e1b61e1ce71df3`。

脚本会打印下载进度，检查每个文件 SHA256。重试时复用已完成且校验通过的文件；**单个没下完的 `.part` 文件会重新下载，不支持该文件的断点续传**。网络经常断开时可使用 ModelScope 官方客户端下载同一 revision 到新目录，再核验文件；不要混合不同版本。

模型目录中的 `eval/` 是发布者的历史结果，不能复制到你的实验目录冒充本次运行结果。

### 5.5 PaliGemma tokenizer 授权和下载

π0.5 的预处理器指定 `google/paligemma-3b-pt-224` tokenizer。**老师提供模型权重，不代表你自动获得此依赖的访问权限。** 若 Hugging Face 返回 401/403，先在该模型页面按要求接受许可并申请访问，再在自己的终端执行 `hf auth login`。访问令牌只在交互登录中输入，不粘贴到实验记录、Agent 提示词或日志。

下载 tokenizer 等小文件，不下载 PaliGemma 大权重：

```bash
python - <<'PY'
import json, os
from pathlib import Path
from huggingface_hub import HfApi, snapshot_download
repo = 'google/paligemma-3b-pt-224'
revision = HfApi().model_info(repo).sha
path = snapshot_download(repo, revision=revision, local_dir=os.environ['PI_DEP'],
                         allow_patterns=['*.json', '*.model', '*.txt', '*.jinja'])
Path(os.environ['PI_DEP'], 'dependency_revision.json').write_text(
    json.dumps({'repo':repo, 'revision':revision}, indent=2)+'\n')
print(path, revision)
PY
```

这里先解析并保存当前依赖 commit；后续复现使用记录的 commit，不再重新挑选最新版。

### 5.6 创建 π0.5 部署副本

```bash
python "$TOOLS/prepare_checkpoint.py" \
  --source "$PI_RAW" --dest "$PI" --dependency "$PI_DEP" \
  2>&1 | tee "$RUN/logs/prepare_pi05.log"
```

π0.5 的几处接口不能照抄 SmolVLA：

| 项目 | SmolVLA 发布模型 | π0.5 发布模型 |
|---|---|---|
| 环境实际状态输入 | 取左臂 8 维 | 双臂 16 维 |
| 网络动作输出 | 左臂 8 维，右臂保持 | 双臂 16 维 |
| 相机字段 | front / left_wrist / right_wrist | processor 映射到 base_0_rgb / left_wrist_0_rgb / right_wrist_0_rgb |
| 状态和动作归一化 | MEAN_STD | QUANTILES，使用 checkpoint 自带 q01/q99 |
| 本文起始执行 horizon | 50 | 50 |
| 本文起始 flow 推理步数 | 25 | 10 |

π0.5 `config.json` 的状态 shape 写了 32，但发布数据和归一化统计量都是 16 维。配套适配器输入真实 16 维关节状态，让原 processor 按发布统计量处理；**不要因为看到 32 就自己拼上速度，也不要在归一化前随意补零。**

模型内部把图像处理到所需分辨率；环境仍提供原始 RGB 三视角。不要把 OpenCV BGR 图像未经转换地喂给模型。

π0.5 使用模型卡中的完整训练指令；配套脚本固定该句并记录到 `run.json` 和回合 metadata。**不能直接沿用 SmolVLA 的简短指令作为 π0.5 基线。** 发布者报告短改写可能导致完全不动作。

### 5.7 π0.5 冒烟和一个完整预算回合

```bash
python "$TOOLS/evaluate_teacher.py" \
  --model pi05 --checkpoint "$PI" --device cuda \
  --horizon 50 --flow-steps 10 --inference-seed 1000 \
  --episodes 1 --seed-start 21999 --max-steps 20 \
  --out "$RUN/pi05_smoke20" \
  2>&1 | tee "$RUN/logs/pi05_smoke20.log"
```

先确认权重严格加载和 20 步成功运行，再执行：

```bash
python "$TOOLS/evaluate_teacher.py" \
  --model pi05 --checkpoint "$PI" --device cuda \
  --horizon 50 --flow-steps 10 --inference-seed 1000 \
  --episodes 1 --seed-start 22000 --max-steps 500 --diagnostic \
  --out "$RUN/pi05_demo500" \
  2>&1 | tee "$RUN/logs/pi05_demo500.log"
python "$TOOLS/summarize_results.py" "$RUN/pi05_demo500/result.json"
```

特别说明：公开 LeRobot 0.5.1 的 `PI05Policy.from_pretrained` 会捕获一些权重加载错误并返回模型。配套适配器改用显式构造、沿用其 key 映射、`load_state_dict(strict=True)`；加载不完整就停止。**这不是提升性能的算法修改，而是防止把随机初始化当已加载模型。**

本文按发布配置的 bfloat16 模式部署；发布者模型卡中部分历史评测使用 FP32、900 步、seed 0–19。本文 500 步的新评测是你自己的统一协议，不能称为已经复现其 4.05/10。若要复现原数字，还需核对同一源码、场景、精度、指令和全部参数。

## 6. 多回合评测：先固定协议

### 6.1 本文采用独立的新 seed 区间

| 集合 | seeds | 用途 |
|---|---|---|
| 接口冒烟 | 21999 | 20 步检查，不计成功率 |
| 开发集 | 22000–22004 | 找问题、选参数，5 回合 |
| 确认集 | 22100–22119 | 检查候选修改是否可重复，20 回合；仍属开发 |
| 最终集 | 23000–23019 | 配置固定后验收，20 回合，不反馈调参 |

这与之前 ACT 指南的区间分开，避免把已看过的旧结果当全新验收。若要和 ACT 直接比较，请把 ACT 最终候选在本文相同场景协议下重跑，而不是拿不同 seeds、步数的旧表直接拼接。

这些 seeds 是你的评测留出，不保证与发布者的全部训练经历绝对无重叠。老师 SmolVLA 的完整 200 回合训练集未在本次核查中取得；不能宣称已经证明严格的训练数据去重。

固定条件：2×5、basic、来源第 1 列、盒和饼干随机化开启、最多 500 控制步、workers=1、原始成功判定。两个模型使用各自适配的正确指令和相应归一化。

### 6.2 先跑专家对照

```bash
python "$TOOLS/evaluate_teacher.py" \
  --model expert --episodes 5 --seed-start 22000 --max-steps 500 \
  --out "$RUN/expert_dev5" \
  2>&1 | tee "$RUN/logs/expert_dev5.log"
python "$TOOLS/summarize_results.py" "$RUN/expert_dev5/result.json"
```

专家使用仿真真值，作为流程/任务参考，不是与纯视觉模型相同输入条件的公平学习算法对手。专家不需要 GPU 模型推理，但仍需要仿真环境。

### 6.3 两个模型各跑 5 个开发场景

在 SmolVLA 对应环境及机器中：

```bash
python "$TOOLS/evaluate_teacher.py" \
  --model smolvla --checkpoint "$SMOL" --device cuda \
  --horizon 50 --flow-steps 25 --inference-seed 1000 \
  --episodes 5 --seed-start 22000 --max-steps 500 \
  --out "$RUN/smol_h50_dev5" \
  2>&1 | tee "$RUN/logs/smol_h50_dev5.log"
python "$TOOLS/summarize_results.py" "$RUN/smol_h50_dev5/result.json"
```

在 π0.5 对应环境及机器中：

```bash
python "$TOOLS/evaluate_teacher.py" \
  --model pi05 --checkpoint "$PI" --device cuda \
  --horizon 50 --flow-steps 10 --inference-seed 1000 \
  --episodes 5 --seed-start 22000 --max-steps 500 \
  --out "$RUN/pi05_h50_dev5" \
  2>&1 | tee "$RUN/logs/pi05_h50_dev5.log"
python "$TOOLS/summarize_results.py" "$RUN/pi05_h50_dev5/result.json"
```

每个运行目录：

```text
run.json                 实际参数、环境、源码信息、模型哈希
episodes.jsonl           每完成一个回合立即追加，防止中断后全丢
result.json              全部回合完成才生成的正式报告
analysis_summary.json    汇总脚本生成的精简事实
episode_scores.csv       逐回合分数、成功与否、步数、时间
error.json               若运行异常则记录；不能当任务失败分数
trace/                   仅 --diagnostic 时生成
```

汇总脚本会检查回合数量、seed 唯一性及区间。正常完整评测有 5/5 回合；崩溃只完成 3 回合不能拿 3 回合冒充 5 回合。不自动续跑原目录；重试使用新目录，保留旧错误记录，最终只比较完整的同协议运行。

### 6.4 如何读结果

- `mean_final_cookies`：每回合结束时平均入盒数量，满分 10。
- `successful_episodes / episodes`：完成全部任务的回合比例。
- `success_wilson95`：成功率的 Wilson 95% 区间。样本少区间会宽；它不是平均入盒数的置信区间。
- `distribution`：例如多少回合停在 0/5/10 个，帮助选诊断样本。
- `failed_seeds`：用于重放失败回合。
- `failure_reason`：环境记录的原因；不等于模型内部推理过程。
- `policy_call_timing`：包含预处理、取动作和后处理，混有“新预测动作块”和“从队列取动作”的调用，不能称为每次完整网络前向的延迟。

不同 GPU 上的墙钟时间不能直接比较模型速度。视频录制也显著拖慢运行。首次比较先看同场景任务成绩，速度实验需同硬件、无录制、明确计时范围。

## 7. 让 Agent 参与：两种用途，分别留证据

此处 Agent 指有工具调用能力的编码/实验助手。机械臂动作仍由 SmolVLA 或 π0.5 产生；Agent 不应把仿真真值塞进模型输入来“帮助成功”。

### 用途 A：部署和接口审计 Agent

你可以把下面这段作为发给编码 Agent 的任务。不是告诉老师“我问了一个问题”就结束，要保存它实际检查文件、运行检查、定位问题的过程。

```text
请协助我检查 A3 老师发布模型的部署。
先读取本指南、配套 prepare_checkpoint.py、teacher_policy.py、evaluate_teacher.py，
以及实际的 config.json、processor JSON、deployment_manifest.json 和部署日志。

检查：模型类型、8D/16D 动作、相机键、RGB 处理、归一化来源、tokenizer 路径、
指令、严格权重加载、Python/LeRobot/Torch 版本、GPU 显存。
先做低成本配置/导入检查；真实模型运行只做 20 控制步冒烟，不启动训练。
不要修改老师原始权重、任务成功判定或 ACT 环境。
若发现问题，先定位到具体文件和报错，再提出最小修改，保留修改前后差异。
禁止以 strict=False 或跳过归一化来掩盖不兼容。
输出一份 deployment_review.md：已证实事实、证据路径、未解决问题、下一条命令。
不得将“配置通过”写成“模型成功率已验证”。
```

保留：Agent 对话、工具日志、部署前后 diff、成功的冒烟报告、失败过的错误日志。访问令牌和账号不进入材料。

### 用途 B：基于证据的实验 Agent

先运行第 6 节。给 Agent 的输入应是实际 `analysis_summary.json`、CSV、`run.json`，以及代表性失败视频/trace。不要一次贴入整份巨大的 trace。

```text
请作为实验分析 Agent，分析本轮 A3 SmolVLA/π0.5 的开发集结果。
先读我提供的 run.json、analysis_summary.json 和 episode_scores.csv，核对完整性。
如需解释失败，再选 1–2 个失败 seed，读取视频或对应 trace 的相关时间段。

任务：
1. 区分环境/加载故障和正常执行后的任务失败。
2. 输出“观察事实 → 证据文件与时间/步号 → 假设 → 还缺的证据”。
3. 只提出一个可验证的部署参数修改，并说明为什么该修改针对观察到的问题。
4. 给出与基线配对的运行命令，保持权重、场景、seed、评分和步数预算相同。
5. 结果出来后比较逐 seed 的差异，再决定保留、撤销或继续收集证据。

本轮限制：只允许改变 horizon，候选范围为 50、8、4；最多评估两个新候选，
每个候选先跑开发集 22000–22004；不训练、不换 checkpoint、不改变判定器，
不访问最终集 23000–23019。最多使用 2 小时 GPU 时间；达到预算时保留真实部分记录，
不能补造完整结果。需要继续实验时由我决定是否增加预算。
你可以运行上述范围内的评测工具；不得下载额外大模型或扩大任务范围。
输出 experiment_decision.md，并区分未执行建议与已经执行的结果。
```

如果 Agent 只能聊天、不能执行命令，就采用“Agent 分析并给命令 → 你执行 → 将结果反馈给 Agent”的人工协作模式，并在报告中如实说明。不能称为全自动 Agent。

Agent 的价值要从它是否正确检查接口、是否根据证据定位问题、是否减少无效试验来展示，而不只统计它生成了多少字。

## 8. 一个具体实验：执行动作块长度是否影响失败

假设你观察到：50 步动作执行期间，物体已经偏移，但机械臂还在按旧预测继续移动。此时可以提出：**缩短执行 horizon，是否能更频繁地利用新画面纠偏？** 这是待验证假设，不保证提升。

50 步在 20 Hz 下相当于 2.5 秒后重新规划；8 步约 0.4 秒；4 步约 0.2 秒。预测块仍保持 50，只执行前一部分就重新预测，计算调用也会增加。

例如 Agent 根据证据选择 SmolVLA 的 horizon=8：

```bash
python "$TOOLS/evaluate_teacher.py" \
  --model smolvla --checkpoint "$SMOL" --device cuda \
  --horizon 8 --flow-steps 25 --inference-seed 1000 \
  --episodes 5 --seed-start 22000 --max-steps 500 \
  --out "$RUN/smol_h8_dev5" \
  2>&1 | tee "$RUN/logs/smol_h8_dev5.log"
python "$TOOLS/summarize_results.py" "$RUN/smol_h8_dev5/result.json"
```

π0.5 对应实验：

```bash
python "$TOOLS/evaluate_teacher.py" \
  --model pi05 --checkpoint "$PI" --device cuda \
  --horizon 8 --flow-steps 10 --inference-seed 1000 \
  --episodes 5 --seed-start 22000 --max-steps 500 \
  --out "$RUN/pi05_h8_dev5" \
  2>&1 | tee "$RUN/logs/pi05_h8_dev5.log"
python "$TOOLS/summarize_results.py" "$RUN/pi05_h8_dev5/result.json"
```

**这两条只是候选命令，不是必须全部运行的自动扫描。** 先读证据，再决定是否运行。预先把所有组合跑一遍叫参数扫描，不能据此声称 Agent 自主发现了最优策略。

若基线完全不动作或全是 NaN，不应先扫描 horizon；回到权重、指令和归一化检查。若只有第二批始终没开始，缩短 horizon 也未必解决，需要视频验证并记录无效假设。

### 8.1 如何确认，而不是只看 5 回合的小波动

对基线 50 和一个候选 8，在新确认场景各跑 20 回合。下面以 SmolVLA 为例；如果选择 π0.5，改模型/路径并将 flow 步数固定为 10。

```bash
for H in 50 8; do
  python "$TOOLS/evaluate_teacher.py" \
    --model smolvla --checkpoint "$SMOL" --device cuda \
    --horizon "$H" --flow-steps 25 --inference-seed 1000 \
    --episodes 20 --seed-start 22100 --max-steps 500 \
    --out "$RUN/smol_h${H}_confirm20" \
    2>&1 | tee "$RUN/logs/smol_h${H}_confirm20.log" || break
  python "$TOOLS/summarize_results.py" "$RUN/smol_h${H}_confirm20/result.json" || break
done
```

逐 seed 比较“候选最终入盒数 − 基线最终入盒数”，统计改善、持平、变差各多少回合；再看完整成功回合数和耗时。不要仅挑选候选成功的那几个 seed 展示。

可以直接使用配套比较工具：

```bash
python "$TOOLS/compare_runs.py" \
  --baseline "$RUN/smol_h50_confirm20/result.json" \
  --candidate "$RUN/smol_h8_confirm20/result.json" \
  --out "$RUN/smol_h50_vs_h8_confirm20.json"
```

输出里的 `improved/tied/worse` 是入盒数改善/持平/变差的回合数；`mean_cookie_delta` 是平均每回合多搬了几个。工具检查模型、权重、seed、预算和主要判定条件一致，条件不一致就报错。它不替你做显著性结论，源码快照是否一致也需要核对。

如果效果不稳定，就保留基线并如实写“未确认改善”。期中项目中的有价值结果也包括有证据的失败分析。

### 8.2 最终验收

确认集之后写下所选参数，再访问最终集。以最终选中 horizon=8 为例；若实际选中其他值，改成真实值。

```bash
python "$TOOLS/evaluate_teacher.py" \
  --model smolvla --checkpoint "$SMOL" --device cuda \
  --horizon 8 --flow-steps 25 --inference-seed 1000 \
  --episodes 20 --seed-start 23000 --max-steps 500 \
  --out "$RUN/smol_selected_final20" \
  2>&1 | tee "$RUN/logs/smol_selected_final20.log"
python "$TOOLS/summarize_results.py" "$RUN/smol_selected_final20/result.json"
```

如果要声称“比基线提升”，还要在相同最终集跑基线 horizon=50，另存 `smol_baseline_final20`。不要用最终集再次挑选参数；若看了结果后又调参，就承认这个集合已进入开发，并另设新最终集。

若只想比较两个发布模型，则先固定各自配置，再分别跑同一最终集。结论应是“这两个发布 checkpoint 在本协议下的表现”，不能推出“架构 A 普遍优于 B”，因为训练数据与训练预算不同。

## 9. 失败回合重放与日志阅读

先在 `analysis_summary.json` 找 `failed_seeds`。下面示例 22002 必须替换为你真实失败的 seed，不代表已观测到该 seed 失败。

```bash
python "$TOOLS/evaluate_teacher.py" \
  --model smolvla --checkpoint "$SMOL" --device cuda \
  --horizon 50 --flow-steps 25 --inference-seed 1000 \
  --episodes 1 --seed-start 22002 --max-steps 500 --diagnostic \
  --out "$RUN/smol_failure_seed22002" \
  2>&1 | tee "$RUN/logs/smol_failure_seed22002.log"
```

重放时所有策略参数必须和原失败运行一致，否则是新实验。相同推理 seed 提高复现性，但未启用所有 GPU 确定性选项，不能保证所有硬件逐位相同；若重放不同，也记录这种波动。

日志的用途：

| 证据 | 能回答什么 | 不能直接证明什么 |
|---|---|---|
| 最终数量 | 最后搬进去多少 | 失败原因 |
| 视频 | 是否抓空、掉落、停住、返回 | 模型内部为什么做决定 |
| 原始动作/实际动作 | 模型命令是否有限、环境是否限幅 | 单凭夹爪数值就确认抓取成功 |
| 仿真诊断真值 | 事后确认饼干位置和接触 | 不能把这些真值当视觉策略的输入 |

给每个诊断写简短记录：`seed / 观察步段 / 视频时间 / 可确认现象 / 假设 / 待验证项`。不要把规则专家的阶段名称硬套成神经网络的真实内部阶段。

## 10. 实验记录模板与汇报内容

每次修改都新建一份 Markdown 记录，使用下面模板：

```text
实验编号：
父实验目录：
Agent 使用方式：部署审计 / 实验分析 / 工具执行
模型来源与 revision：
权重 SHA256：
场景源码快照 / commit：
实际任务指令：
问题与证据（文件、seed、步号）：
Agent 假设：
唯一修改：
其余固定条件：
完整命令：
是否真的执行：
实际回合数 / 预期回合数：
最终平均入盒数：
完整成功回合数 / 总数：
与父实验逐 seed 比较：
运行成本和硬件：
结论：保留 / 撤销 / 证据不足
下一步及理由：
```

最后汇报至少包含：

1. 资源关系图：示范数据、基座、任务权重、仿真评测各是什么。
2. 两个模型输入输出对比，以及真实部署成功的日志证据。
3. 明确的评测协议、完整逐回合分数及成功率。
4. 一段典型成功或失败视频；失败也能用于说明问题。
5. 两种 Agent 用法的实际记录：接口审计与实验分析/决策。
6. 至少一组基线与单因素候选的对照，或如实说明为何尚未进行。
7. 还未解决的问题，例如 π0.5 的资源限制、某个配置兼容错误、样本量有限。

不得把发布者的评测、Agent 的预测、未运行命令的预期效果填写成自己的实测结果。

## 11. 常见问题与下一步

| 问题 | 怎么处理 |
|---|---|
| `No module named transformers` / `PI05Policy` | 确认激活本模型环境，安装对应 `[smolvla]` / `[pi]` extra，查看 `which python` |
| `/home/eter/...` 或 `/root/autodl-tmp/...` 不存在 | 使用部署副本，检查 prepare 日志，不创建假的发布者目录 |
| unknown config key | 只能处理已审阅的兼容字段；保留报错，不一键删除所有未知项 |
| 权重参数缺失/形状不符 | 停止性能评测，核对发布 checkpoint 与实现版本，不跳过严格加载 |
| tokenizer 401/403 | 检查该依赖访问许可和交互登录；与 CUDA 无关 |
| `Expected ... size 16/32` | 检查原始状态、processor stats 和映射顺序，不凭 shape 猜含义 |
| π0.5 OOM | 8 GB 不适合本路线；换大显存环境，确认其他进程占用；不要改模型层数硬塞 |
| SmolVLA OOM | 检查是否有其他训练进程，单模型、单 worker 运行；缩短 horizon 不缩小模型本身 |
| EGL 渲染错误 | 先重跑原基础相机检查，检查 GPU/EGL；不是模型学习效果差 |
| 20 步 success=false | 正常，先看是否完整执行和动作是否有效 |
| `result.json` 缺失 | 读 error/log；属于运行未完成，不计算完整成功率 |
| 目录已存在 | 新命名重跑目录，保留之前记录 |
| 5 回合全成功 | 是开发集 5/5，不代表所有场景 100% |
| 一个任务好、另一个任务差 | 分开报告，2×5 到 2×10 是迁移，不能混成一个成功率 |

本指南刻意先完成 2×5。basic 稳定后可以在同一脚本增加 `--profile medium` 或 `advanced`，每档单独保存并报告；不要在不同档位间混合分数。2×10 需要额外核对任务指令和接入，当前配套脚本不支持，因此没有提供一个看似通用但未适配的 2×10 命令。

## 12. 今天建议完成到哪里

- 第一段：完成第 2–3 节，拿到独立环境和部署清单。
- 第二段：完成第 4 节，有严格加载日志、20 步报告和一段真实视频。
- 第三段：完成 SmolVLA 的 5 回合开发评测及 Agent 用途 A/B 的首次记录。
- 大显存机器就绪后再完成第 5 节 π0.5，沿用相同评测协议。
- 最后才做候选参数实验、确认集和最终集；ACT 按原指南独立推进。

### 12.1 关掉终端后，如何继续

本地 SmolVLA 新终端：先重复第 2.1 节恢复变量，再执行：

```bash
source ~/.venvs/a3-teacher-smol/bin/activate
which python
printf 'checkpoint=%s\nrun=%s\n' "$SMOL" "$RUN"
test -f "$SMOL/deployment_manifest.json"
```

服务器 π0.5 新终端：重复第 5.2 节，再执行：

```bash
source ~/.venvs/a3-teacher-pi05/bin/activate
which python
printf 'checkpoint=%s\nrun=%s\n' "$PI" "$RUN"
test -f "$PI/deployment_manifest.json"
```

然后继续尚未运行的实验。不要重新执行已完成的 `--out` 目录；如果上次中断需要重跑，把目录名和日志名一起改为例如 `smol_h50_dev5_retry01`。

### 12.2 怎样把结果交给 Agent，或发给我看

优先提供下面这些小文件，而不是把整个模型、数据集或 trace 打包：

1. 正常运行：`run.json`、`analysis_summary.json`、`episode_scores.csv`。
2. 报错运行：`error.json`、日志末尾约 80 行，以及你实际执行的完整命令。
3. 解释行为：选一个失败 seed 的 `front.mp4`；必要时再加手腕视角或特定 trace 步段。
4. 比较实验：基线和候选的汇总、配对比较 JSON、Agent 的实验假设记录。

查看日志末尾的例子：

```bash
tail -n 80 "$RUN/logs/smol_h50_dev5.log"
```

Windows 本地结果可以从资源管理器打开 `D:\School\Agent\project\a3_dual_arm_sim\outputs\teacher_models_01`。服务器结果先下载到一个标明模型和服务器的文件夹，避免与本地同名结果相互覆盖。

### 12.3 暂时不要做的重复工作

本阶段直接部署任务权重，不需要先采集数据，也不需要跑 `train-smolvla` 或重新训练 π0.5。不要为了凑“完整流程”多跑一次几万步训练；本阶段完整性体现在加载、评测、分析、验证和可追溯记录上。

## 13. 核查来源

以下来源用于核对接口和发布内容；网页中的安装建议不是本次已经执行过的操作。

- [A3 发布页](https://github.com/Eter0109/a3_dual_arm_sim)
- [固定上游版本的策略适配接口](https://github.com/Eter0109/a3_dual_arm_sim/blob/08ebe8cb370856051d0b32b07e691cc9b056716b/src/a3_dual_arm_sim/workflows/policy_adapters.py)
- [SmolVLA 发布 checkpoint](https://huggingface.co/Eter0109/a3-cookie-multi-left8-40k-v1/tree/ffa4dfbc3e36796a4b02bcf07e6776f6077e49b8)
- [SmolVLA 训练和数据清单](https://huggingface.co/Eter0109/a3-cookie-multi-left8-40k-v1/blob/ffa4dfbc3e36796a4b02bcf07e6776f6077e49b8/a3_inference_manifest.json)
- [π0.5 发布模型与模型卡](https://www.modelscope.cn/models/yunsx114/a3-dual-arm-cookie-pi05-aligned)
- [π0.5 训练数据元数据](https://www.modelscope.cn/api/v1/datasets/yunsx114/a3-dual-arm-cookie-main-aligned/repo?Revision=master&FilePath=meta/info.json)
- [LeRobot 0.5.1 发布包](https://pypi.org/project/lerobot/0.5.1/)，本次按 wheel 源码核对配置、processor、模型加载器。
- [PaliGemma tokenizer 所属模型](https://huggingface.co/google/paligemma-3b-pt-224)

附注：本文保持旧指南的实验记录与开发/最终集分离原则；硬件、当前源码、模型配置与依赖源码在本次重新核查。具体成功率必须等待你实际运行后填写。
