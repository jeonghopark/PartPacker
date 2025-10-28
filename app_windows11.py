import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
import argparse
from datetime import datetime

import cv2
import gradio as gr
import kiui
import numpy as np
import rembg
import torch
import torch.nn as nn
import trimesh

try:
    # running on Hugging Face Spaces
    import spaces
except ImportError:
    # running locally, use a dummy space
    class spaces:
        class GPU:
            def __init__(self, duration=60):
                self.duration = duration

            def __call__(self, func):
                return func

# download checkpoints
from huggingface_hub import hf_hub_download

from flow.configs.schema import ModelConfig
from flow.model import Model
from flow.utils import get_random_color, recenter_foreground
from vae.utils import postprocess_mesh

flow_ckpt_path = hf_hub_download(repo_id="nvidia/PartPacker", filename="flow.pt")
vae_ckpt_path = hf_hub_download(repo_id="nvidia/PartPacker", filename="vae.pt")

TRIMESH_GLB_EXPORT = np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]]).astype(np.float32)
MAX_SEED = np.iinfo(np.int32).max
bg_remover = rembg.new_session()

# model config
model_config = ModelConfig(
    vae_conf="vae.configs.part_woenc",
    vae_ckpt_path=vae_ckpt_path,
    qknorm=True,
    qknorm_type="RMSNorm",
    use_pos_embed=False,
    dino_model="dinov2_vitg14",
    hidden_dim=1536,
    flow_shift=3.0,
    logitnorm_mean=1.0,
    logitnorm_std=1.0,
    latent_size=4096,
    use_parts=True,
)

# Multi-GPU setup
def setup_multi_gpu():
    """Configures multiple GPUs and assigns devices."""
    if not torch.cuda.is_available():
        return {'primary': 'cpu', 'secondary': 'cpu', 'num_gpus': 0}
    
    num_gpus = torch.cuda.device_count()
    print(f"Number of available GPUs: {num_gpus}")
    
    if num_gpus >= 2:
        # Separate flow and VAE models if 2 or more GPUs are available
        primary_device = 'cuda:0'  # For flow model
        secondary_device = 'cuda:1'  # For VAE model
        print(f"Enabling model parallelism: Flow -> {primary_device}, VAE -> {secondary_device}")
    else:
        # Single GPU case
        primary_device = 'cuda:0'
        secondary_device = 'cuda:0'
        print(f"Using single GPU: {primary_device}")
    
    return {
        'primary': primary_device,
        'secondary': secondary_device, 
        'num_gpus': num_gpus
    }


def _dtype_for(device_str: str):
    """Return optimal dtype for device"""
    if device_str.startswith("cuda"):
        return torch.bfloat16
    return torch.float32


class MultiGPUModel(nn.Module):
    """Model wrapper for multi-GPU support."""
    def __init__(self, model_config, gpu_config):
        super().__init__()
        self.gpu_config = gpu_config
        self.config = model_config

        self.primary = gpu_config.get('primary', 'cpu')
        self.secondary = gpu_config.get('secondary', self.primary)

        self.dtype_primary = _dtype_for(self.primary)
        self.dtype_secondary = _dtype_for(self.secondary)

        # base model
        self.base_model = Model(model_config).eval()

        # Flow → primary
        if hasattr(self.base_model, 'flow'):
            self.base_model.flow = self.base_model.flow.to(
                device=self.primary, dtype=self.dtype_primary
            )

        # VAE → secondary
        if hasattr(self.base_model, 'vae'):
            self.base_model.vae = self.base_model.vae.to(
                device=self.secondary, dtype=self.dtype_secondary
            )

        # Other modules → primary
        for name, module in self.base_model.named_children():
            if name not in ['flow', 'vae']:
                module.to(device=self.primary, dtype=self.dtype_primary)

    def forward(self, data, num_steps=50, cfg_scale=7):
        # Move input tensors to primary device/dtype
        for k, v in data.items():
            if torch.is_tensor(v):
                data[k] = v.to(device=self.primary, dtype=self.dtype_primary)

        if self.gpu_config.get('num_gpus', 0) > 0 and torch.cuda.is_available():
            torch.cuda.empty_cache()

        with torch.inference_mode():
            return self.base_model(data, num_steps=num_steps, cfg_scale=cfg_scale)

    def vae_decode(self, data, resolution=384):
        # Move VAE input to secondary device/dtype
        for k, v in data.items():
            if torch.is_tensor(v):
                data[k] = v.to(device=self.secondary, dtype=self.dtype_secondary)

        if self.gpu_config.get('num_gpus', 0) > 0 and torch.cuda.is_available():
            torch.cuda.empty_cache()

        with torch.inference_mode():
            return self.base_model.vae(data, resolution=resolution)


# Parse command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--multi', action='store_true', help='Enable multi-GPU support')
args = parser.parse_args()

# Initialize GPU configuration and model based on arguments
if args.multi:
    gpu_config = setup_multi_gpu()
    model = MultiGPUModel(model_config, gpu_config)
    multi_gpu_enabled = True
    print("=" * 60)
    print("Multi-GPU mode enabled")
    print(f"Primary device (Flow): {gpu_config['primary']}")
    print(f"Secondary device (VAE): {gpu_config['secondary']}")
    print("=" * 60)
else:
    gpu_config = {'num_gpus': 1 if torch.cuda.is_available() else 0}
    
    if torch.cuda.is_available():
        device = "cuda"
        dtype = torch.bfloat16
        print("=" * 60)
        print(f"CUDA device detected: {torch.cuda.get_device_name(0)}")
        print(f"Using dtype: {dtype}")
        print(f"CUDA version: {torch.version.cuda}")
        print("=" * 60)
    else:
        device = "cpu"
        dtype = torch.float32
        print("=" * 60)
        print("WARNING: CUDA not available, using CPU")
        print("Performance will be significantly slower")
        print("=" * 60)

    model = Model(model_config).eval()
    model = model.to(device=device, dtype=dtype)
    
    multi_gpu_enabled = False

# Load checkpoint
print(f"Loading flow checkpoint...")
ckpt_dict = torch.load(flow_ckpt_path, weights_only=True, map_location=device if not multi_gpu_enabled else 'cpu')

if multi_gpu_enabled:
    model.base_model.load_state_dict(ckpt_dict, strict=True)
else:
    model.load_state_dict(ckpt_dict, strict=True)

print("Model loaded successfully!")

# Get random seed
def get_random_seed(randomize_seed, seed):
    if randomize_seed:
        seed = np.random.randint(0, MAX_SEED)
    return seed


# Process image
@spaces.GPU(duration=10)
def process_image(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if image.shape[-1] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGBA)
    else:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # bg removal if there is no alpha channel
        image = rembg.remove(image, session=bg_remover)  # [H, W, 4]
    mask = image[..., -1] > 0
    image = recenter_foreground(image, mask, border_ratio=0.1)
    image = cv2.resize(image, (518, 518), interpolation=cv2.INTER_AREA)
    return image


# Process 3D generation
@spaces.GPU(duration=90)
def process_3d(
    input_image, num_steps=50, cfg_scale=7, grid_res=384, seed=42, 
    simplify_mesh=False, target_num_faces=100000
):
    # Seed
    kiui.seed_everything(seed)
    
    # Display GPU memory usage
    if multi_gpu_enabled and gpu_config['num_gpus'] > 0:
        for i in range(gpu_config['num_gpus']):
            memory_allocated = torch.cuda.memory_allocated(i) / 1024**3
            memory_reserved = torch.cuda.memory_reserved(i) / 1024**3
            print(f"GPU {i}: Allocated {memory_allocated:.2f}GB, Reserved {memory_reserved:.2f}GB")
    elif torch.cuda.is_available():
        memory_allocated = torch.cuda.memory_allocated(0) / 1024**3
        memory_reserved = torch.cuda.memory_reserved(0) / 1024**3
        print(f"GPU: Allocated {memory_allocated:.2f}GB, Reserved {memory_reserved:.2f}GB")

    # Output path
    os.makedirs("output", exist_ok=True)
    output_glb_path = f"output/partpacker_{datetime.now().strftime('%Y%m%d_%H%M%S')}.glb"

    # Process input image
    image = input_image.astype(np.float32) / 255.0
    image = image[..., :3] * image[..., 3:4] + (1 - image[..., 3:4])  # white background
    
    # Create tensor
    if multi_gpu_enabled:
        target_device = model.primary if hasattr(model, 'primary') else 'cuda'
        target_dtype = model.dtype_primary if hasattr(model, 'dtype_primary') else torch.bfloat16
    else:
        target_device = device
        target_dtype = dtype
    
    image_tensor = (
        torch.from_numpy(image)
        .permute(2, 0, 1)
        .contiguous()
        .unsqueeze(0)
        .to(device=target_device, dtype=target_dtype)
    )

    data = {"cond_images": image_tensor}

    print(f"\n{'='*60}")
    print(f"Starting generation:")
    print(f"  Steps: {num_steps} | CFG: {cfg_scale} | Resolution: {grid_res}")
    if torch.cuda.is_available():
        est_time_min = num_steps * 0.5 / 60  # Rough estimate for CUDA
        est_time_max = num_steps * 1.0 / 60
        print(f"  Estimated time: {est_time_min:.1f}-{est_time_max:.1f} minutes")
    print(f"{'='*60}\n")

    if multi_gpu_enabled:
        # Multi-GPU processing
        results = model(data, num_steps=num_steps, cfg_scale=cfg_scale)
        latent = results["latent"]

        # Query mesh - process each part separately
        data_part0 = {"latent": latent[:, : model.config.latent_size, :]}
        data_part1 = {"latent": latent[:, model.config.latent_size :, :]}

        # Generate mesh for part 0
        results_part0 = model.vae_decode(data_part0, resolution=grid_res)
        torch.cuda.empty_cache()
        
        # Generate mesh for part 1
        results_part1 = model.vae_decode(data_part1, resolution=grid_res)
        torch.cuda.empty_cache()
    else:
        # Single GPU processing
        with torch.inference_mode():
            results = model(data, num_steps=num_steps, cfg_scale=cfg_scale)

        latent = results["latent"]

        # Query mesh
        data_part0 = {"latent": latent[:, : model.config.latent_size, :]}
        data_part1 = {"latent": latent[:, model.config.latent_size :, :]}

        with torch.inference_mode():
            results_part0 = model.vae(data_part0, resolution=grid_res)
            results_part1 = model.vae(data_part1, resolution=grid_res)

    if not simplify_mesh:
        target_num_faces = -1

    # Process part 0
    vertices, faces = results_part0["meshes"][0]
    mesh_part0 = trimesh.Trimesh(vertices, faces)
    mesh_part0.vertices = mesh_part0.vertices @ TRIMESH_GLB_EXPORT.T
    mesh_part0 = postprocess_mesh(mesh_part0, target_num_faces)
    parts = mesh_part0.split(only_watertight=False)

    # Process part 1
    vertices, faces = results_part1["meshes"][0]
    mesh_part1 = trimesh.Trimesh(vertices, faces)
    mesh_part1.vertices = mesh_part1.vertices @ TRIMESH_GLB_EXPORT.T
    mesh_part1 = postprocess_mesh(mesh_part1, target_num_faces)
    parts.extend(mesh_part1.split(only_watertight=False))

    # Filter small parts
    parts = [part for part in parts if len(part.faces) > 10]

    # Assign colors to parts
    for j, part in enumerate(parts):
        part.visual.vertex_colors = get_random_color(j, use_float=True)

    mesh = trimesh.Scene(parts)
    mesh.export(output_glb_path)

    print(f"\n✓ Generation complete! Saved to: {output_glb_path}\n")

    return output_glb_path


# Gradio UI
if multi_gpu_enabled:
    _TITLE = """PartPacker: Efficient Part-level 3D Object Generation (Multi-GPU Mode)"""
    _DESCRIPTION = f"""
<div>
<a style="display:inline-block" href="https://research.nvidia.com/labs/dir/partpacker/"><img src='https://img.shields.io/badge/public_website-8A2BE2'></a>
<a style="display:inline-block; margin-left: .5em" href="https://github.com/NVlabs/PartPacker"><img src='https://img.shields.io/github/stars/NVlabs/PartPacker?style=social'/></a>
</div>

**Multi-GPU Configuration:**
* {gpu_config['num_gpus']} GPUs detected
* Flow model on {gpu_config['primary']}
* VAE model on {gpu_config['secondary']}

**Usage:**
* Each part is visualized with a random color and can be separated in the GLB file
* Try different random seeds if output is unsatisfactory
* Reduce Grid resolution if running out of memory
"""
else:
    _TITLE = """PartPacker: Efficient Part-level 3D Object Generation"""
    device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    _DESCRIPTION = f"""
<div>
<a style="display:inline-block" href="https://research.nvidia.com/labs/dir/partpacker/"><img src='https://img.shields.io/badge/public_website-8A2BE2'></a>
<a style="display:inline-block; margin-left: .5em" href="https://github.com/NVlabs/PartPacker"><img src='https://img.shields.io/github/stars/NVlabs/PartPacker?style=social'/></a>
</div>

**System:** {device_name} | **Dtype:** {dtype}

**Usage:**
* Each part is visualized with a random color and can be separated in the GLB file
* Try different random seeds if output is unsatisfactory
* Recommended: 25-50 steps, CFG 5-7, Resolution 256-384
"""

block = gr.Blocks(title=_TITLE).queue()
with block:
    with gr.Row():
        with gr.Column():
            gr.Markdown("# " + _TITLE)
    gr.Markdown(_DESCRIPTION)

    with gr.Row():
        with gr.Column(scale=1):
            with gr.Row():
                input_image = gr.Image(label="Input Image", type="filepath")
                seg_image = gr.Image(label="Processed Image", type="numpy", interactive=False, image_mode="RGBA")
            
            with gr.Accordion("Settings", open=True):
                # Quick presets
                with gr.Row():
                    preset_fast = gr.Button("⚡ Fast (1-2 min)", size="sm")
                    preset_balanced = gr.Button("⚖️ Balanced (2-4 min)", size="sm")
                    preset_quality = gr.Button("🎨 Quality (5-10 min)", size="sm")
                
                # Inference steps
                num_steps = gr.Slider(
                    label="Inference steps", 
                    minimum=1, maximum=100, step=1, value=50,
                    info="More steps = better quality but slower"
                )
                
                # CFG scale
                cfg_scale = gr.Slider(
                    label="CFG scale", 
                    minimum=2, maximum=10, step=0.1, value=7.0,
                    info="Higher = more faithful to input"
                )
                
                # Grid resolution
                default_grid_res = 256 if multi_gpu_enabled else 384
                min_grid_res = 192 if multi_gpu_enabled else 256
                input_grid_res = gr.Slider(
                    label="Grid resolution", 
                    minimum=min_grid_res, maximum=512, step=16, 
                    value=default_grid_res,
                    info="Higher = more detail but requires more memory"
                )
                
                # Random seed
                with gr.Row():
                    randomize_seed = gr.Checkbox(label="Randomize seed", value=True)
                    seed = gr.Slider(label="Seed", minimum=0, maximum=MAX_SEED, step=1, value=0)
                
                # Simplify mesh
                with gr.Row():
                    simplify_mesh = gr.Checkbox(label="Simplify mesh", value=False)
                    target_num_faces = gr.Slider(
                        label="Target face count", 
                        minimum=10000, maximum=1000000, step=1000, value=100000
                    )
                
                # Generate button
                button_gen = gr.Button("🚀 Generate 3D Model", variant="primary")

        with gr.Column(scale=1):
            output_model = gr.Model3D(label="Generated 3D Model", height=512)

    with gr.Row():
        gr.Examples(
            examples=[
                ["assets/images/rabbit.png"],
                ["assets/images/robot.png"],
                ["assets/images/teapot.png"],
                ["assets/images/barrel.png"],
                ["assets/images/cactus.png"],
                ["assets/images/cyan_car.png"],
                ["assets/images/pickup.png"],
                ["assets/images/swivelchair.png"],
                ["assets/images/warhammer.png"],
            ],
            fn=process_image,
            inputs=[input_image],
            outputs=[seg_image],
            cache_examples=False,
        )

    # Preset button actions
    preset_fast.click(
        lambda: (25, 5.0, 256),
        outputs=[num_steps, cfg_scale, input_grid_res]
    )
    preset_balanced.click(
        lambda: (35, 6.0, 320),
        outputs=[num_steps, cfg_scale, input_grid_res]
    )
    preset_quality.click(
        lambda: (50, 7.5, 384),
        outputs=[num_steps, cfg_scale, input_grid_res]
    )

    # Main generation pipeline
    button_gen.click(
        process_image, 
        inputs=[input_image], 
        outputs=[seg_image]
    ).then(
        get_random_seed, 
        inputs=[randomize_seed, seed], 
        outputs=[seed]
    ).then(
        process_3d,
        inputs=[seg_image, num_steps, cfg_scale, input_grid_res, seed, simplify_mesh, target_num_faces],
        outputs=[output_model],
    )

if __name__ == "__main__":
    import webbrowser
    import time
    from threading import Timer
    
    def open_browser():
        webbrowser.open('http://127.0.0.1:7860')
    
    print("\n" + "="*60)
    print("Starting PartPacker WebUI...")
    print("If browser doesn't open automatically, visit:")
    print("http://127.0.0.1:7860")
    print("="*60 + "\n")
    
    # 2초 후 브라우저 자동 열기
    Timer(2, open_browser).start()
    
    # share=True로 localhost 문제 우회
    block.queue().launch(
        share=True,
        show_error=True,
        quiet=False
    )
