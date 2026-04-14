python train.py -s data/denseview/scan1 -o output/denseview/scan1 --sfm_config posed --use_view_config --config_view_num 20  --tetra_downsample_ratio 0.25 --use_dense_view

```bash
CUDA_VISIBLE_DEVICES=1 python train.py \
-s data/denseview/scan1 \
-o output/denseview/scan1 \
--sfm_config posed \
--use_view_config \
--config_view_num 20 \
--tetra_downsample_ratio 0.25 \
--use_dense_view
```


```bash

pixi run python scripts/generate_view_split.py \
  -s /autodl-fs/data/fastgs/nt7_sr \
  -train 0,168,169,170,171,172,173,174,175,176,177,178,179,312,313,314,315,316,317,318,319,320,321,322,323 \
  --dense_view_divisor 2 \
  --dense_view_output /autodl-fs/data/fastgs/nt7_sr/dense_view.json

# 12 视角交错序列时，--dense_view_divisor 2 表示保留 0-11，丢 12-23，再保留 24-35 这种整组抽样。

pixi run python scripts/generate_view_split.py \
  -s /autodl-fs/data/fastgs/dm1_sr \
  -train 0,168,169,170,171,172,173,174,175,176,177,178,179,312,313,314,315,316,317,318,319,320,321,322,323 \
  --dense_view_divisor 2 \
  --dense_view_output /autodl-fs/data/fastgs/dm1_sr/dense_view.json

```
  --depth_model geometrycrafter \
  --geometry_prior_mode hybrid-override-at-align-prep

```bash
CUDA_VISIBLE_DEVICES=0 pixi run python train.py \
-s /autodl-fs/data/fastgs/nt1_sr \
-o data/g4/nt1_sr_sm9 \
--sfm_config posed \
--use_view_config \
--config_view_num 25 \
--tetra_downsample_ratio 0.25 \
--use_dense_view \
--resolution 2 \
--merge_device cuda \
--merge_resolution_scale 1 \
--depth_model geometrycrafter \
--geometry_prior_mode hybrid-override-at-align-prep \
--checkpoint_iterations 18000 \
--free_gaussians_config dense_compact_sm1_d   --dense_regul weak



CUDA_VISIBLE_DEVICES=1 pixi run python train.py \
-s /autodl-fs/data/fastgs/nt6_sr \
-o data/g4/nt6_sr_sm9 \
--sfm_config posed \
--use_view_config \
--config_view_num 25 \
--tetra_downsample_ratio 0.25 \
--use_dense_view \
--resolution 2 \
--merge_device cuda \
--merge_resolution_scale 1 \
--depth_model geometrycrafter \
--geometry_prior_mode hybrid-override-at-align-prep \
--free_gaussians_config dense_compact_sm9   --dense_regul weak





CUDA_VISIBLE_DEVICES=1 scripts/continue_dense_view_stage.sh \
    data/g4/nt3_sr/mast3r_sfm \
    data/g4/nt3_sr/free_gaussians \
    --config dense_compact_sm2 \
    --resolution 1 \
    --merge-resolution-scale 2 \
    --merge-device cuda


```
CUDA_VISIBLE_DEVICES=1 /root/autodl-tmp/home/rais/G4Splat/scripts/continue_dense_view_stage.sh \
    /autodl-fs/data/g4/nt4_sr/mast3r_sfm \
    /autodl-fs/data/g4/nt4_sr/free_gaussians \
    --config dense_compact_sm \
    --resolution 2 \
    --merge-device cuda \
    --merge-resolution-scale 2 \
    --tetra-downsample-ratio 0.25

### 重跑 dense-view stage 第二阶段


  CUDA_VISIBLE_DEVICES=1 scripts/rerun_dense_view_stage.sh \
    /autodl-fs/data/g4/nt4_sr \
    --config dense_compact

  如果以后想换配置或迭代数，也可以：

  CUDA_VISIBLE_DEVICES=1 scripts/rerun_dense_view_stage.sh \
    output/nt/nt1_step3_interleaved_exhaustive \
    --config dense_compact \
    --iteration 7000

CUDA_VISIBLE_DEVICES=1 scripts/rerun_dense_view_stage.sh \
    output/nt/nt1_step3_interleaved_exhaustive \
    --config dense_compact_sm


    -------
    ## 处理反射

    - scripts/find_bottom_dominant_plane.py

  它会对每一张 plane_mask_frame*.npy：

  - 在下半部分找 label != 0 里像素数最多的区域
  - 输出报告
  - 可生成预览图
  - 可把该区域对应的整块 plane 置 0
  - 会自动备份原始 .npy

  ### 直接用

  先只看、不改：

  pixi run python scripts/find_bottom_dominant_plane.py \
    --plane_root_path /autodl-fs/data/g4/dm7_sr_v2/mast3r_sfm/plane-refine-depths \
    --report-json /autodl-fs/data/g4/dm7_sr_v2/mast3r_sfm/plane-refine-depths/bottom_plane_report.json \
    --preview-dir /autodl-fs/data/g4/dm7_sr_v2/mast3r_sfm/plane-refine-depths/bottom_plane_previews

  确认效果后，真正删除：

  pixi run python scripts/find_bottom_dominant_plane.py \
    --plane_root_path /autodl-fs/data/g4/dm7_sr_v2/mast3r_sfm/plane-refine-depths \
    --apply-zero \
    --report-json /autodl-fs/data/g4/dm7_sr_v2/mast3r_sfm/plane-refine-depths/bottom_plane_report.json \
    --preview-dir /autodl-fs/data/g4/dm7_sr_v2/mast3r_sfm/plane-refine-depths/bottom_plane_previews

  ### 可调参数

  - --lower-start-ratio 0.5
    默认下半部分；比如 0.6 表示更靠底部 40%
  - --min-bottom-pixels 1000
    太小的区域不删

  ### 删完之后

  再重跑：

  pixi run python scripts/plane_refine_depth.py \
    --source_path /autodl-fs/data/g4/dm7_sr_v2/mast3r_sfm \
    --plane_root_path /autodl-fs/data/g4/dm7_sr_v2/mast3r_sfm/plane-refine-depths \
    --pnts_path /autodl-fs/data/g4/dm7_sr_v2/mast3r_sfm/chart_pcd.ply

  然后再重跑 refine_free_gaussians.py。


相机轨迹

- /autodl-fs/data/g4/nt4_sr/free_gaussians/train/ours_7000/renders/00000_camera_trajectory.json
  - /autodl-fs/data/g4/nt4_sr/free_gaussians/train/ours_7000/renders/00000_camera_trajectory_unity.json

  我还补了可复用脚本：

  - scripts/export_g4_render_camera.py
