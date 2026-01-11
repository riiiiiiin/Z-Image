# Z-Image 轨迹实验（可控初始噪声 / 保存每步结果）

这份文档说明如何在 Z-Image native 实现中：

1. 找到“VAE latent 初始噪声采样 + transformer 预测 + scheduler 一步步 denoise + VAE decode”的代码位置；
2. 通过 hook 保存每一步的 latents（以及可选的中间 decode 图像）；
3. 用不同初始噪声分布（normal / uniform 等）对比整条去噪轨迹。

## 代码位置（你要找的那段）

- 初始 latent 噪声采样发生在：
  - [src/zimage/pipeline.py](src/zimage/pipeline.py) 的 `generate()` 内部，创建 `shape` 后生成 `latents`。
- 一步一步 denoise 的循环发生在：
  - [src/zimage/pipeline.py](src/zimage/pipeline.py) 的 `for i, t in timesteps` 循环内。
  - 每一步会调用 transformer 预测 `noise_pred`，然后 `scheduler.step(...)` 得到新的 `latents`。
- 最终 decode 成图片发生在：
  - [src/zimage/pipeline.py](src/zimage/pipeline.py) 的 `_decode_latents_to_pil()`。

## 思路（复述 + 完善）

目标：固定除了“初始 VAE latent 噪声”以外的所有因素（prompt、scheduler、steps、guidance、seed、模型权重等），只替换 `latents` 的初始化方式，然后观察：

- 每一步 `latents` 张量如何变化；
-（可选）每一步 decode 成图片后，视觉轨迹如何变化；
- 最终输出图片的变化。

实现方式：

1. 在 `generate()` 内部把初始噪声采样做成“可注入”的：
   - 默认仍然是 `torch.randn(...)`（保持现有行为不变）
   - 允许传入 `noise_sampler(shape, generator, device, dtype)` 或直接传入 `initial_latents`
2. 在 denoise 循环末尾（`scheduler.step` 后）加 callback hook：
   - 每一步把 `latents`（以及可选的 decoded 图）作为 payload 传出去
   - 外部用一个 `TrajectoryRecorder` 回调，把每一步保存到磁盘
3. 用脚本批量跑不同噪声分布：
   - `normal`（高斯）
   - `uniform`（均匀）
   - 其它分布可以自己实现一个新的 `noise_sampler` 并传入

## 新增模块

- [src/zimage/noise.py](src/zimage/noise.py)
  - `normal_noise_sampler()` / `uniform_noise_sampler()`
  - 可直接传给 `generate(noise_sampler=...)`

- [src/zimage/trajectory.py](src/zimage/trajectory.py)
  - `TrajectoryRecorder`：一个 callback，负责把每一步保存为：
    - `latents/step_XXXX.pt`
    - `images/step_XXXX_00.png`（需要开启 decode）
    - `images/final_00.png`
    - `trajectory.json`（元数据 + 每步索引）

## 直接运行：轨迹实验脚本

脚本：
- [trajectory_experiment.py](trajectory_experiment.py)

安装依赖（仓库根目录）：

```bash
pip install -e .
```

### 1) 正态噪声（默认）

```bash
python trajectory_experiment.py \
  --noise normal \
  --seed 42 \
  --out runs/cat_normal
```

### 2) 均匀噪声（其它不变）

```bash
python trajectory_experiment.py \
  --noise uniform \
  --uniform-low -1 \
  --uniform-high 1 \
  --seed 42 \
  --out runs/cat_uniform
```

### 3) 保存每步 decode 图像（更直观，但更慢）

```bash
python trajectory_experiment.py \

  --noise normal \
  --decode \
  --decode-every 1 \
  --out runs/cat_normal_decode
```

## 输出目录结构

以 `--out runs/cat_normal` 为例：

- `runs/cat_normal/trajectory.json`
- `runs/cat_normal/latents/step_0000.pt`
- `runs/cat_normal/latents/step_0001.pt`
- ...
- `runs/cat_normal/latents/final.pt`
- `runs/cat_normal/images/step_0000_00.png`（开启 `--decode` 才会有）
- `runs/cat_normal/images/final_00.png`

## 自定义噪声分布（扩展）

你可以自己写一个函数：

- 输入：`shape, generator, device, dtype`
- 输出：一个 `torch.Tensor`，形状必须等于 `shape`

然后传给 `generate(noise_sampler=...)`。
