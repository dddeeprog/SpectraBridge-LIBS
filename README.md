# S1（统一版）｜单模态光谱域自适应 + 先分类再定量 Soft Routing

> 目标：将**煤饼（源域）**训练好的“先分类再定量”模型，稳健迁移到**煤粉（目标域）**现场；在实时性约束下，兼顾准确度与可解释性。

---

## 0. 快速概览（What & How）

* **共享编码器（SpecEncoder）**：Conv1d→Transformer→全局表征 *z*。
* **两条头**：

  * 分类头（ClassifierHead）输出 `logits`（置信度/类别）。
  * 回归头：

    * **每类回归** `y_per_class ∈ R[C]`（针对不同煤种的专属回归）
    * **全局回归** `y_global`（类别不可分/低置信时的兜底）
* **软路由（Soft Routing）**：`y_soft = Σ softmax(logits)_c · y_per_class_c`；若 `max softmax < τ` 则回退到 `y_global`；可选 EMA 平滑。
* **迁移学习**：

  * **Pretrain**：源域监督训练。
  * **UDA**：无监督目标域（Deep CORAL 二阶统计对齐）。
  * **SSDA**：少量目标标注的微调（可选）。

---

## 1. 目录结构

```
S1/
├─ mgtl/
│  ├─ __init__.py
│  ├─ data.py
│  ├─ transforms.py
│  ├─ losses/
│  │  ├─ __init__.py
│  │  ├─ coral.py
│  │  └─ focal.py
│  ├─ models/
│  │  ├─ __init__.py
│  │  ├─ spec_encoder.py
│  │  ├─ classifier.py
│  │  └─ regressors.py
│  └─ utils/
│     ├─ __init__.py
│     ├─ metrics.py
│     ├─ seed.py
│     └─ logging.py
├─ config.yaml
├─ train.py
├─ eval.py
├─ infer.py
├─ export_onnx.py
└─ README.md（本文档）
```

> 说明：各 `__init__.py` 已就位，支持 `from mgtl import ...` 的统一导入。

---

## 2. 环境与安装

* **Python** ≥ 3.9
* **PyTorch**（CUDA 建议）
* 其他：`numpy`, `pyyaml`；导出 ONNX 需 `onnx`, `onnxsim`（可选）

示例（conda）：

```bash
conda create -n mgtl python=3.10 -y
conda activate mgtl
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121  # 依据你的 CUDA 版本
pip install numpy pyyaml onnx onnxsim
```

---

## 3. 数据准备（CSV → NPY）

### 3.1 标准输入格式（NPY/NPZ）

* `X.npy`：形状 `[N, L]` 的 **float32** 光谱矩阵（每行一条光谱，长度 `L=Spectral_length`）。
* `y.npy`（可选）：形状 `[N]` 的 **float32** 回归标签。
* `c.npy`（可选）：形状 `[N]` 的 **int64** 类别标签（`0..C-1`）。

### 3.2 为什么推荐先转 NPY，而不是直接读 CSV？

* **解析与 IO**：CSV 行文本解析 + 类型转换成本高；NPY 是二进制，**零解析**、加载快（支持 `mmap`）。
* **一致性**：转换阶段可强制统一长度 `L`（插值/裁剪）、去 NaN/Inf、统一 dtype，避免训练期隐性异常。
* **可扩展**：大数据直接 `mmap`，无需一次性全部载入内存。
* **稳定性**：避免现场 CSV 多样化表头/分隔符/小数点等问题；训练/推理口径完全一致。

> 结论：工业现场**值得**多一步转换（fast, safe, consistent）。

### 3.3 转换示例

使用你项目根目录下的 `csv_to_npy.py`（已在早前提供；不再在脚本内写长注释，说明迁回本 README）：

```bash
python csv_to_npy.py \
  --csv_dir data_raw/source --out_dir data/source \
  --spectral_length 2048 --has_y --has_c

python csv_to_npy.py \
  --csv_dir data_raw/target --out_dir data/target \
  --spectral_length 2048 --has_y --has_c=false
```

---

## 4. 配置（`S1/config.yaml`）关键项

* `spectral_length`: 光谱长度 `L`（训练/评估/推理/导出一致）
* `n_classes`: 煤种数量。
* `paths.*`: 源/目标/产物目录、`class_map.json`（可选）。
* `model.*`: 编码器/Transformer/池化等结构超参。
* `train.*`: stage（`pretrain|uda|ssda`）、batch/epochs/AMP/早停、优化器/调度器、冻结策略、损失权重（`ce/softmix/per_class/global/coral`）、分类不平衡（Focal）。
* `infer.*`: `tau`（低置信回退阈值）、`alpha`（EMA 平滑系数）、`fallback`（标记）。
* `export.*`: ONNX opset/动态轴/简化/是否 fp16（注：fp16 多由引擎侧处理）。

---

## 5. 一键运行（Train / Eval / Infer / Export）

> 以下命令均以 `S1/` 为工作目录示例。

### 5.1 训练（包含 UDA / SSDA）

```bash
# 1) 仅源域监督预训练（Pretrain）
sed -i "s/stage:.*/stage: pretrain/" S1/config.yaml
python S1/train.py --config S1/config.yaml --save artifacts/best_pretrain.pt

# 2) 无监督域自适应（UDA）
sed -i "s/stage:.*/stage: uda/" S1/config.yaml
python S1/train.py --config S1/config.yaml \
  --resume artifacts/best_pretrain.pt --save artifacts/best_uda.pt

# 3) 半监督域自适应（SSDA，可选）
sed -i "s/stage:.*/stage: ssda/" S1/config.yaml
python S1/train.py --config S1/config.yaml \
  --resume artifacts/best_uda.pt --save artifacts/best_ssda.pt
```

> 复现性：已接入 `seed_worker/get_generator`，DataLoader 可复现；需确保相同 `seed` 与相同数据顺序。

### 5.2 评估（Eval）

```bash
python S1/eval.py --config S1/config.yaml \
  --ckpt artifacts/best_uda.pt \
  --domains source target \
  --out artifacts/eval.json
```

输出包含：

* `reg_soft`：软路由回归口径的 RMSE/MAE/MAPE/R²
* `reg_global`：全局回归口径
* `cls`：Accuracy / Precision / Recall / F1（macro/weighted）

### 5.3 推理（Infer）

（保持原节，已放在文末《推理（infer.py）》部分，含 JSON 指标/回退/EMA 平滑说明。）

### 5.4 导出（Export ONNX）

**导出命令**（与 `S1/export_onnx.py` 一致）

```bash
python S1/export_onnx.py \
  --config S1/config.yaml \
  --ckpt artifacts/best_uda.pt \
  --out artifacts/model.onnx \
  --opset 17 \
  --dynamic   # 动态 batch 维（可去掉以固定 batch，提高某些引擎性能）
```

> 备注：`config.export.fp16` 仅作为占位标记；真正的 FP16/INT8 常在推理引擎侧完成（TensorRT/ONNX Runtime EP 等）。

**输入/输出签名**（按 `export_onnx.py` 约定）

* **输入**：`x: float32[B, L]`，L=`config.spectral_length`
* **输出**：

  * `logits: float32[B, C]` —— 分类分数（softmax 由引擎端完成）
  * `y_per_class: float32[B, C]` —— 每类回归值
  * `y_global: float32[B]` —— 全局回归值
  * `y_soft: float32[B]` —— 软加权回归（已在图内计算 `Σ softmax(logits)*y_per_class`）

**动态轴说明**

* `--dynamic` 时，导出包含 `batch` 动态维：`x` 的维度 0、各输出的维度 0 均标记为动态；
* 若你的线上服务使用固定批（如 256/512），可不加 `--dynamic`，让引擎做更激进的优化（尤其是 TensorRT）。

**推理端路由与平滑（τ 回退 + EMA）示例**（ONNX Runtime，Python）：

```python
import numpy as np, onnxruntime as ort
sess = ort.InferenceSession("artifacts/model.onnx", providers=["CUDAExecutionProvider","CPUExecutionProvider"])

def infer_batch(x_np, tau=0.6, alpha=0.2, ema_state=None):
    # x_np: float32 [B, L]
    outs = sess.run(["logits","y_per_class","y_global","y_soft"], {"x": x_np})
    logits, y_pc, y_g, y_soft = [o.astype(np.float32) for o in outs]

    # 低置信回退：max softmax < tau → 用 y_global
    p = np.exp(logits - logits.max(axis=1, keepdims=True))
    p = p / p.sum(axis=1, keepdims=True)
    conf = p.max(axis=1)
    y_hat = np.where(conf >= tau, y_soft, y_g)

    # EMA 指数平滑：按 **时间/到达顺序** 做串行平滑
    if alpha > 0:
        if ema_state is None:
            ema_state = float(y_hat[0])
            y_hat[0] = ema_state
            idx = 1
        else:
            idx = 0
        for i in range(idx, len(y_hat)):
            ema_state = alpha * float(y_hat[i]) + (1.0 - alpha) * ema_state
            y_hat[i] = ema_state
    return y_hat, conf, ema_state
```

> 部署要点：**EMA 是有状态**的（per-stream/per-conveyor）。若多条皮带并发或分段处理，请为每条/每段维护独立 `ema_state`。

**τ/α 标定建议**

1. 在目标域验证集上网格搜索：`tau ∈ [0.4, 0.85]`、`alpha ∈ {0, 0.1, 0.2, 0.3}`；
2. 指标以 `RMSE` 为主、兼顾 `MAPE` 与极值误差（95th/99th 百分位），记录 `reg_hat` 与 `reg_soft/reg_global`；
3. 一般经验：

   * 分类置信较稳：`tau≈0.6~0.7`、`alpha≈0.1~0.3` 可平抑抖动；
   * 分类尚不稳定：先取 **更高 τ**（多用 `y_global` 兜底），`alpha` 先设 0，再逐步加大。

**性能与稳定性建议**

* **ONNX Runtime**：

  * GPU：`CUDAExecutionProvider` + IO Binding（`io_binding = sess.io_binding()`）减少拷贝；
  * CPU：可尝试动态量化（`ort.quantization.quantize_dynamic`）在不牺牲太多精度的前提下降低延迟；
* **TensorRT**：

  * 若固定批、固定长度，优先静态导出；
  * FP16/INT8 需在构建 engine 时开启（INT8 需校准集）；
* **多线程/批处理**：

  * 在线系统可采用小批（如 32/64）攒批推理，结合流式 EMA；
  * 注意按“到达顺序”恢复 EMA，避免乱序导致的平滑失效。

**与训练前处理的一致性**

* 训练时若用了样本级标准化（`data.normalize: standard`），推理输入也应保持相同口径；
* 若在线端在接入层做了额外归一化/滤波，请确保与训练时的 `transforms` 对齐，或在导出前将预处理融入模型（本 S1 仅做最小预处理，推荐在数据接入层完成）。

**输出使用的两种口径**

* **Soft**：使用导出图内的 `y_soft`，配合 τ 回退与 EMA（推荐）；
* **Global-only**：直接取 `y_global`，避免分类依赖，适合早期上线或应急兜底。

---

## 6. 损失与路由设计（可调配方）

* **分类**：`CE` 或 `FocalLoss(γ, α)`（应对类不平衡）。
* **回归**：

  * **软加权** `MSE(y_soft, y)`（默认权重 `softmix_weight`）
  * **每类回归** `MSE(y_{c*}, y)`（仅当有真实类别 `c*`；`per_class_reg_weight`）
  * **全局回归** `MSE(y_global, y)`（`global_reg_weight`）
* **域对齐**：`Deep CORAL(z_s, z_t)`（`coral_weight` + `mean_align_weight` + `normalize`）
* **口径**：

  * `profile=soft`：先分类再定量（低置信回退 `tau` → `y_global`）。
  * `profile=global_only`：纯全局回归（不依赖分类）。

---

## 7. 推理（infer.py）

**输入要求**

* `X.npy`（必需）：形状 `[N, L]`，`L` 与 `config.yaml -> spectral_length` 一致
* `y.npy`（可选）：形状 `[N]`，若存在将计算回归指标
* `c.npy`（可选）：形状 `[N]`，若存在将计算分类指标
* `class_map.json`（可选）：类别索引与名称映射，支持数组 `["烟煤", ...]` 或字典 `{"0":"烟煤",...}`

**一键推理命令（含低置信回退与 EMA 平滑 + 指标 JSON）**

```bash
python S1/infer.py \
  --config S1/config.yaml \
  --ckpt artifacts/best.pt \
  --input_dir data/target \
  --out artifacts/infer_target.csv \
  --json_out artifacts/infer_target.json \
  --alpha 0.2 \
  --tau 0.6 \
  --profile soft \
  --batch_size 256 \
  --num_workers 2 \
  --pin_memory
```

**口径与参数说明**

* `--profile`: `soft`（默认，先分类再定量的软加权）｜`global_only`（仅全局回归，不走分类路由）
* `--tau`: **低置信回退阈值**。当 `max softmax(logits) < tau` 时回退到 `y_global`
* `--alpha`: **EMA 指数平滑系数**（按样本顺序），`0` 表示关闭
* `--json_out`: 若 `input_dir` 中存在 `y.npy` / `c.npy`，将额外写出回归/分类指标到该 JSON
* 输出 CSV 含列：`idx, y_global, y_soft, y_hat, cls_pred, cls_conf, [cls_name], [y_true], [abs_err_*], [c_true], [c_name]`
* 保持输入顺序不打乱（推理内部已固定 `shuffle=False` 且接入统一随机种子）

---

## 8. 常见问题（FAQ）

* **Q：加载时提示 `X.npy` 形状不为 `[N, L]`？**
  A：需保证每条光谱长度等于 `spectral_length`；使用转换脚本插值/裁剪到一致长度。

* **Q：类别不平衡影响分类？**
  A：启用 `train.classification.use_focal=true` 并设置 `focal_gamma`；或在数据层做重采样。

* **Q：现场波长微漂导致性能下降？**
  A：模型内已加入一定稳健性；若漂移较大，建议在线波长校准或在训练中使用 `RandomWavelengthShift` 增广（`mgtl/transforms.py`）。

* **Q：导出 ONNX 后怎么做 `tau` 回退与 EMA？**
  A：ONNX 输出 `logits/y_per_class/y_global/y_soft`，在推理引擎侧实现 `max_softmax<tau → y_global` 与 EMA 即可（示例见 `infer.py`）。

* **Q：为什么不直接用 CSV？**
  A：见上文 3.2；NPY 更快更稳、易于 `mmap` 与一致性控制，是工业在线更可靠的上游格式。

---

## 9. 版本与致谢

* S1 统一版：`v1.0`（本仓）
* 主要模块：`SpecEncoder / ClassifierHead / PerClassRegressors / GlobalRegressor / Deep CORAL / FocalLoss`
* 指标：`RMSE/MAE/MAPE/R² + ACC/P/R/F1 + 混淆矩阵`