# Chart And Depth Artifacts

这份文档解释 `mast3r_sfm` 目录下几类最容易混淆的中间产物，重点说明：

- `render-charts-train-views/charts_conf_frame*.png`
- `render-charts-train-views/depth_frame*.{png,tiff}`
- `render-charts-train-views/mono_depth_frame*.{png,tiff}`
- `plane-refine-depths/refine_depth_frame*.tiff`
- `plane-refine-depths/confident_map_frame*.png`

本文以这个真实场景作为例子：

```text
/autodl-fs/data/g4/nt6_sm8-1/mast3r_sfm
```

## TL;DR

- `charts_conf_frame*.png` 不是 RGB，也不是 MASt3R 直接导出的“原始置信度图”。
- 它是 `align_charts` 之后保存在 `charts_data.npz` 里的 chart confidence，经插值放大和伪彩色可视化后得到的图。
- 它看起来“一格一格”是正常现象，因为它本质上是低分辨率 chart/pointmap 上的置信度场，不是自然图像纹理。
- `mono_depth_frame*.tiff` 更接近“单目先验深度”。
- `depth_frame*.tiff` 更接近“chart 对齐后的深度”。
- `refine_depth_frame*.tiff` 是进一步结合平面约束后得到的深度，通常更接近后续训练实际使用的深度。

## 目录之间的关系

常见链路是：

```text
MASt3R scene
  -> align_charts
  -> charts_data.npz
  -> render-charts-train-views/*
  -> plane_refine_depth
  -> plane-refine-depths/*
```

更具体一点：

1. `scripts/align_charts.py` 读取 `mast3r_sfm` 场景。
2. 它结合 MASt3R 场景、SFM 对齐信息和 DepthAnything 深度先验，运行 chart alignment。
3. alignment 结果保存到 `charts_data.npz`。
4. `2d-gaussian-splatting/render_chart_views.py` 把 `charts_data.npz` 可视化成 `render-charts-train-views/`。
5. `scripts/plane_refine_depth.py` 再调用平面相关脚本，把 chart depth 进一步 refine 到 `plane-refine-depths/`。

## 每种文件到底是什么

### `charts_conf_frame000000.png`

路径示例：

```text
/autodl-fs/data/g4/nt6_sm8-1/mast3r_sfm/render-charts-train-views/charts_conf_frame000000.png
```

它表示当前视角下的 chart confidence 可视化图。

直接生成位置在 [render_chart_views.py](/root/autodl-tmp/home/rais/G4Splat/2d-gaussian-splatting/render_chart_views.py)，代码会把：

- `charts_priors["confs"]`

取出来，然后：

```python
plt.imsave(..., vis_charts_conf, cmap='viridis')
```

也就是说：

- 它是一个单通道浮点 confidence map 的伪彩色 PNG。
- PNG 本身是可视化，不是原始数值载体。
- 真正的原始数值来自 `charts_data.npz` 里的 `confs`。

### 它是不是 MASt3R 生成的

不完全是。

更准确地说：

- 上游场景和部分约束来自 MASt3R。
- 但 `charts_conf_frame*.png` 对应的 `confs` 不是 MASt3R 原始 confidence 直接另存为 PNG。
- 它来自 chart alignment 阶段的 learnable confidence。

在 [charts_alignment.py](/root/autodl-tmp/home/rais/G4Splat/matcha/dm_trainers/charts_alignment.py) 里，保存 `charts_data.npz` 时写入了：

- `prior_depths`
- `depths`
- `pts`
- `confs`

而 `confs` 的来源是 `ParallelAligner.confidence`，定义在 [parallel_aligner.py](/root/autodl-tmp/home/rais/G4Splat/matcha/dm_scene/parallel_aligner.py)：

```python
return 1. + torch.exp(self._confidence)
```

所以它更像：

- “chart alignment 优化后学到的置信度”

而不是：

- “MASt3R 原图级置信度的直接导出图”

### 为什么看起来都是方块

这通常是正常现象，主要有两个原因。

第一，它不是照片，而是结构化几何场。

- confidence 在大片平面区域内往往变化比较平缓。
- 这种图天生就比 RGB 更容易显得块状。

第二，它原本分辨率就比训练视图低。

在这个例子里，`charts_data.npz` 里的原始 `confs` 形状是：

```text
(25, 288, 512)
```

而保存出来的 `charts_conf_frame000000.png` 是：

```text
1280 x 720
```

也就是说，低分辨率 confidence map 会先被插值到训练视图分辨率，再着色成 PNG。这个放大过程会让图更容易呈现块状或分片感。

对应代码在 [charts.py](/root/autodl-tmp/home/rais/G4Splat/matcha/dm_scene/charts.py)：

- `charts_data['confs'][:, None]`
- `torch.nn.functional.interpolate(..., mode="bilinear")`

所以“看起来像方块”通常不是 bug，而是数据类型和分辨率决定的视觉结果。

### 它有什么功能

它的核心作用是表达“哪些 chart 区域更可信”。

主要用途包括：

1. 作为几何构建/筛选时的 mask 依据。
2. 参与 chart 相关的距离、过滤或后续结构约束。
3. 帮助调试 alignment 结果，判断哪些区域 confidence 低。

例如在 [charts.py](/root/autodl-tmp/home/rais/G4Splat/matcha/dm_scene/charts.py) 中，会用：

```python
masks = charts_confs > conf_th
```

来过滤低置信度区域并构建 mesh。

## `mono_depth_frame000000.tiff`

路径示例：

```text
/autodl-fs/data/g4/nt6_sm8-1/mast3r_sfm/render-charts-train-views/mono_depth_frame000000.tiff
```

它表示单目深度先验。

在当前实现里，这个“mono depth”不是随机来的，而是：

- DepthAnythingV2 输出的单目深度
- 再经过与几何/相机体系对齐后的结果

在 [render_chart_views.py](/root/autodl-tmp/home/rais/G4Splat/2d-gaussian-splatting/render_chart_views.py) 中，这个量来自：

- `charts_priors["prior_depths"]`

代码注释写得很直接：

- `depth-anything-v2 prior depth (mono depth + linear alignment)`

所以可以把它理解为：

- “单目先验深度”
- “还没经过 chart 对齐优化后的参考深度”

## `depth_frame000000.tiff`

路径示例：

```text
/autodl-fs/data/g4/nt6_sm8-1/mast3r_sfm/render-charts-train-views/depth_frame000000.tiff
```

它表示 chart alignment 输出的深度。

在 [render_chart_views.py](/root/autodl-tmp/home/rais/G4Splat/2d-gaussian-splatting/render_chart_views.py) 中，它来自：

- `charts_priors["depths"]`

你可以把它理解成：

- “经过 chart 对齐之后的深度”
- “比 mono depth 更接近当前几何优化结果的深度”

它通常是后续平面 refinement 的上游输入之一。

## `refine_depth_frame000000.tiff`

路径示例：

```text
/autodl-fs/data/g4/nt6_sm8-1/mast3r_sfm/plane-refine-depths/refine_depth_frame000000.tiff
```

它表示平面约束进一步 refinement 之后的深度。

这个文件由 [refine_depth_with_planes.py](/root/autodl-tmp/home/rais/G4Splat/2d-gaussian-splatting/planes/refine_depth_with_planes.py) 写出。脚本会读取：

- `depth_frame*.tiff`
- `mono_depth_frame*.tiff`
- 平面分割/合并结果

然后输出：

- `refine_depth_frame*.tiff`

因此它比前面的 `depth_frame*.tiff` 又往后走了一步，通常更偏向：

- “带平面一致性修正后的深度”

如果你关心“最终训练更接近用哪张深度”，多数情况下应该优先关注这个 `refine_depth_frame*.tiff`。

## `confident_map_frame000000.png`

路径示例：

```text
/autodl-fs/data/g4/nt6_sm8-1/mast3r_sfm/plane-refine-depths/confident_map_frame000000.png
```

它来自 `plane_refine_depth` 流程最后一步的 inconsistency/confidence 求解。

入口脚本是 [plane_refine_depth.py](/root/autodl-tmp/home/rais/G4Splat/scripts/plane_refine_depth.py)，它在 refine 完 depth 之后会继续调用：

- `2d-gaussian-splatting/guidance/inconsistence_solver.py`

或者：

- `2d-gaussian-splatting/guidance/plane_inconsistency_solver.py`

所以它的语义更接近：

- “plane-refine-depth 阶段内部对可靠区域的判定”

它和 `charts_conf_frame*.png` 不是同一个概念：

- `charts_conf_frame*` 是 chart alignment 的 confidence。
- `confident_map_frame*` 是 plane refine 阶段的不一致性/可靠性图。

## 三层深度最容易混淆的区别

如果只抓主线，可以这样记：

- `mono_depth_frame*`: 单目先验深度，偏“参考”。
- `depth_frame*`: chart alignment 后深度，偏“几何对齐结果”。
- `refine_depth_frame*`: 再经过平面约束 refinement 的深度，偏“后续使用结果”。

## 一个实用的读图顺序

当你想判断某个场景问题出在哪一层时，建议按这个顺序看：

1. 先看 `rgb_frame*.png`
2. 再看 `mono_depth_frame*.png`
3. 再看 `depth_frame*.png`
4. 再看 `charts_conf_frame*.png`
5. 最后看 `refine_depth_frame*.tiff` 和 `confident_map_frame*.png`

这样比较容易定位问题属于：

- 单目先验本身就不行
- chart alignment 没对齐好
- 平面 refinement 把结果拉坏了
- 或者只是可视化方式让图看起来“不像照片”

## 为什么 `charts_conf` 不应该拿“像不像真实图像”来判断

`charts_conf_frame*.png` 只是一张伪彩色的 confidence 可视化图。

所以这些现象本身不说明它坏了：

- 看起来像方块
- 边界不如 RGB 细
- 某些平面区域颜色大块一致
- 不像真实深度图那样平滑连续

更应该关注的是：

- 低 confidence 区域是否集中在遮挡、边界或不稳定区域
- 高 confidence 区域是否大致覆盖可靠表面
- 最终 `refine_depth` 是否比 `mono_depth` 更合理

## 快速查看浮点 TIFF

像 `refine_depth_frame*.tiff` 这种文件通常是 `float32`，直接用系统图片查看器看不一定直观。

仓库里现在有一个辅助脚本：

[preview_depth_tiff.py](/root/autodl-tmp/home/rais/G4Splat/scripts/preview_depth_tiff.py)

用法：

```bash
cd /root/autodl-tmp/home/rais/G4Splat
pixi run python scripts/preview_depth_tiff.py \
  /autodl-fs/data/g4/nt6_sm8-1/mast3r_sfm/plane-refine-depths/refine_depth_frame000000.tiff
```

默认会在同目录生成：

```text
refine_depth_frame000000_preview.png
```

## 相关代码入口

- chart 对齐入口: [align_charts.py](/root/autodl-tmp/home/rais/G4Splat/scripts/align_charts.py)
- chart 数据保存: [charts_alignment.py](/root/autodl-tmp/home/rais/G4Splat/matcha/dm_trainers/charts_alignment.py)
- chart 数据加载与插值: [charts.py](/root/autodl-tmp/home/rais/G4Splat/matcha/dm_scene/charts.py)
- chart 可视化导出: [render_chart_views.py](/root/autodl-tmp/home/rais/G4Splat/2d-gaussian-splatting/render_chart_views.py)
- plane refine 入口: [plane_refine_depth.py](/root/autodl-tmp/home/rais/G4Splat/scripts/plane_refine_depth.py)
- plane refine 主逻辑: [refine_depth_with_planes.py](/root/autodl-tmp/home/rais/G4Splat/2d-gaussian-splatting/planes/refine_depth_with_planes.py)
- inconsistency/confident map: [inconsistence_solver.py](/root/autodl-tmp/home/rais/G4Splat/2d-gaussian-splatting/guidance/inconsistence_solver.py)

## 结论

如果你看到：

```text
render-charts-train-views/charts_conf_frame000000.png
```

最合适的心智模型是：

- 它是 chart alignment 阶段的 confidence 可视化图。
- 它受 MASt3R 场景影响，但不是 MASt3R 原始 confidence 图本身。
- 它块状是正常现象，尤其在低分辨率 chart 被放大到训练视图分辨率时更明显。
- 真正更值得和最终结果对照的是 `mono_depth_frame*`、`depth_frame*`、`refine_depth_frame*` 这三层深度。
