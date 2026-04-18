
  nt6_sm10：

  cd /root/autodl-tmp/home/rais/G4Splat

  pixi run python scripts/refine_free_gaussians.py \
    --mast3r_scene data/nt6_sm10/mast3r_sfm \
    --output_path data/nt6_sm10/free_gaussians \
    --config dense_compact_sm10 \
    --resolution 2 \
    --dense_regul default \
    --refine_depth_path data/nt6_sm10/mast3r_sfm/plane-refine-depths

  pixi run python 2d-gaussian-splatting/render_multires.py \
    --source_path data/nt6_sm10/mast3r_sfm \
    --model_path data/nt6_sm10/free_gaussians \
    --resolution 2 \
    --skip_test \
    --skip_mesh \
    --render_all_img \
    --use_default_output_dir

  pixi run python scripts/extract_tetra_mesh.py \
    --mast3r_scene data/nt6_sm10/mast3r_sfm \
    --model_path data/nt6_sm10/free_gaussians \
    --output_path data/nt6_sm10/tetra_meshes \
    --config default \
    --downsample_ratio 0.25 \
    --interpolate_views

  nt7_sm10：

  cd /root/autodl-tmp/home/rais/G4Splat

  CUDA_VISIBLE_DEVICES=1 pixi run python scripts/refine_free_gaussians.py \
    --mast3r_scene data/nt7_sm10/mast3r_sfm \
    --output_path data/nt7_sm10/free_gaussians \
    --config dense_compact_sm10 \
    --resolution 2 \
    --dense_regul default \
    --refine_depth_path data/nt7_sm10/mast3r_sfm/plane-refine-depths

    CUDA_VISIBLE_DEVICES=1  pixi run python 2d-gaussian-splatting/render_multires.py \
    --source_path data/nt7_sm10/mast3r_sfm \
    --model_path data/nt7_sm10/free_gaussians \
    --resolution 2 \
    --skip_test \
    --skip_mesh \
    --render_all_img \
    --use_default_output_dir

  pixi run python scripts/extract_tetra_mesh.py \
    --mast3r_scene data/nt7_sm10/mast3r_sfm \
    --model_path data/nt7_sm10/free_gaussians \
    --output_path data/nt7_sm10/tetra_meshes \
    --config default \
    --downsample_ratio 0.25 \
    --interpolate_views
