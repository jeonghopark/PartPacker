import os
from datetime import datetime

import cv2
import gradio as gr
import kiui
import numpy as np
import rembg
import torch
import torch.nn as nn
import trimesh

# メモリ最適化設定
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

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

# マルチGPU設定
def setup_multi_gpu():
    """複数GPUの設定とデバイス割り当て"""
    if not torch.cuda.is_available():
        return {'primary': 'cpu', 'secondary': 'cpu', 'num_gpus': 0}
    
    num_gpus = torch.cuda.device_count()
    print(f"利用可能なGPU数: {num_gpus}")
    
    if num_gpus >= 2:
        # 2つ以上のGPUがある場合、flowモデルとVAEを分離
        primary_device = 'cuda:0'  # flowモデル用
        secondary_device = 'cuda:1'  # VAEモデル用
        print(f"モデル並列化を有効化: Flow -> {primary_device}, VAE -> {secondary_device}")
    else:
        # 1つのGPUしかない場合
        primary_device = 'cuda:0'
        secondary_device = 'cuda:0'
        print(f"単一GPU使用: {primary_device}")
    
    return {
        'primary': primary_device,
        'secondary': secondary_device, 
        'num_gpus': num_gpus
    }

# GPU設定を初期化
gpu_config = setup_multi_gpu()

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

class MultiGPUModel(nn.Module):
    """マルチGPU対応のモデルラッパー"""
    def __init__(self, model_config, gpu_config):
        super().__init__()
        self.gpu_config = gpu_config
        self.config = model_config
        
        # 元のモデルを作成
        self.base_model = Model(model_config).eval()
        
        # flowモデルを最初のGPUに配置
        if hasattr(self.base_model, 'flow'):
            self.base_model.flow = self.base_model.flow.to(gpu_config['primary']).bfloat16()
        
        # VAEモデルを2番目のGPUに配置（利用可能な場合）
        if hasattr(self.base_model, 'vae'):
            self.base_model.vae = self.base_model.vae.to(gpu_config['secondary']).bfloat16()
        
        # その他のコンポーネントを最初のGPUに配置
        for name, module in self.base_model.named_children():
            if name not in ['flow', 'vae']:
                module.to(gpu_config['primary']).bfloat16()
    
    def forward(self, data, num_steps=50, cfg_scale=7):
        """推論実行（デバイス間でのデータ転送を管理）"""
        # 入力データを適切なデバイスに移動
        for key, value in data.items():
            if torch.is_tensor(value):
                data[key] = value.to(self.gpu_config['primary'])
        
        # メモリクリア
        if self.gpu_config['num_gpus'] > 0:
            torch.cuda.empty_cache()
        
        # flowモデルで潜在表現を生成
        with torch.inference_mode():
            results = self.base_model(data, num_steps=num_steps, cfg_scale=cfg_scale)
        
        return results
    
    def vae_decode(self, data, resolution=384):
        """VAEデコード（必要に応じてデバイス間転送）"""
        # VAEのデバイスにデータを移動
        for key, value in data.items():
            if torch.is_tensor(value):
                data[key] = value.to(self.gpu_config['secondary'])
        
        # メモリクリア
        if self.gpu_config['num_gpus'] > 0:
            torch.cuda.empty_cache()
        
        with torch.inference_mode():
            results = self.base_model.vae(data, resolution=resolution)
        
        return results

# マルチGPUモデルを初期化
model = MultiGPUModel(model_config, gpu_config)

# load weight
ckpt_dict = torch.load(flow_ckpt_path, weights_only=True)
model.base_model.load_state_dict(ckpt_dict, strict=True)

# get random seed
def get_random_seed(randomize_seed, seed):
    if randomize_seed:
        seed = np.random.randint(0, MAX_SEED)
    return seed

# process image
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

# process generation
@spaces.GPU(duration=90)
def process_3d(
    input_image, num_steps=50, cfg_scale=7, grid_res=384, seed=42, simplify_mesh=False, target_num_faces=100000
):

    # seed
    kiui.seed_everything(seed)
    
    # GPU使用状況を表示
    if gpu_config['num_gpus'] > 0:
        for i in range(gpu_config['num_gpus']):
            memory_allocated = torch.cuda.memory_allocated(i) / 1024**3
            memory_reserved = torch.cuda.memory_reserved(i) / 1024**3
            print(f"GPU {i}: 使用中メモリ {memory_allocated:.2f}GB, 予約済み {memory_reserved:.2f}GB")

    # output path
    os.makedirs("output", exist_ok=True)
    output_glb_path = f"output/partpacker_{datetime.now().strftime('%Y%m%d_%H%M%S')}.glb"

    # input image (assume processed to RGBA uint8)
    image = input_image.astype(np.float32) / 255.0
    image = image[..., :3] * image[..., 3:4] + (1 - image[..., 3:4])  # white background
    image_tensor = torch.from_numpy(image).permute(2, 0, 1).contiguous().unsqueeze(0).float()

    data = {"cond_images": image_tensor}

    # メモリ最適化のため、より小さなgrid_resに自動調整
    if grid_res > 384 and gpu_config['num_gpus'] < 2:
        print(f"単一GPU使用のため、grid_resを{grid_res}から384に調整")
        grid_res = 384

    # flowモデルで潜在表現を生成
    results = model(data, num_steps=num_steps, cfg_scale=cfg_scale)
    latent = results["latent"]

    # query mesh - 各パートを別々に処理してメモリを節約
    data_part0 = {"latent": latent[:, : model.config.latent_size, :]}
    data_part1 = {"latent": latent[:, model.config.latent_size :, :]}

    # パート0のメッシュ生成
    results_part0 = model.vae_decode(data_part0, resolution=grid_res)
    
    # メモリクリア
    if gpu_config['num_gpus'] > 0:
        torch.cuda.empty_cache()
    
    # パート1のメッシュ生成
    results_part1 = model.vae_decode(data_part1, resolution=grid_res)
    
    # メモリクリア
    if gpu_config['num_gpus'] > 0:
        torch.cuda.empty_cache()

    if not simplify_mesh:
        target_num_faces = -1

    vertices, faces = results_part0["meshes"][0]
    mesh_part0 = trimesh.Trimesh(vertices, faces)
    mesh_part0.vertices = mesh_part0.vertices @ TRIMESH_GLB_EXPORT.T
    mesh_part0 = postprocess_mesh(mesh_part0, target_num_faces)
    parts = mesh_part0.split(only_watertight=False)

    vertices, faces = results_part1["meshes"][0]
    mesh_part1 = trimesh.Trimesh(vertices, faces)
    mesh_part1.vertices = mesh_part1.vertices @ TRIMESH_GLB_EXPORT.T
    mesh_part1 = postprocess_mesh(mesh_part1, target_num_faces)
    parts.extend(mesh_part1.split(only_watertight=False))

    # some parts only have 1 face, seems a problem of trimesh.split.
    parts = [part for part in parts if len(part.faces) > 10]

    # split connected components and assign different colors
    for j, part in enumerate(parts):
        # each component uses a random color
        part.visual.vertex_colors = get_random_color(j, use_float=True)

    mesh = trimesh.Scene(parts)
    # export the whole mesh
    mesh.export(output_glb_path)

    return output_glb_path

# gradio UI
_TITLE = """PartPacker: Efficient Part-level 3D Object Generation via Dual Volume Packing (Multi-GPU対応)"""

_DESCRIPTION = f"""
<div>
<a style="display:inline-block" href="https://research.nvidia.com/labs/dir/partpacker/"><img src='https://img.shields.io/badge/public_website-8A2BE2'></a>
<a style="display:inline-block; margin-left: .5em" href="https://github.com/NVlabs/PartPacker"><img src='https://img.shields.io/github/stars/NVlabs/PartPacker?style=social'/></a>
</div>

* GPU設定: {gpu_config['num_gpus']}基のGPUを検出
* 各パートはランダムな色で可視化され、GLBファイル内で分離できます
* 出力が満足できない場合は、異なるランダムシードを試してください！
* メモリ不足の場合は、Grid resolutionを下げてください
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
                # input image
                input_image = gr.Image(label="Input Image", type="filepath")  # use file_path and load manually
                seg_image = gr.Image(label="Segmentation Result", type="numpy", interactive=False, image_mode="RGBA")
            with gr.Accordion("Settings", open=True):
                # inference steps
                num_steps = gr.Slider(label="Inference steps", minimum=1, maximum=100, step=1, value=50)
                # cfg scale
                cfg_scale = gr.Slider(label="CFG scale", minimum=2, maximum=10, step=0.1, value=7.0)
                # grid resolution - デフォルト値を下げてメモリ使用量を削減
                input_grid_res = gr.Slider(label="Grid resolution", minimum=192, maximum=512, step=1, value=256)
                # random seed
                with gr.Row():
                    randomize_seed = gr.Checkbox(label="Randomize seed", value=True)
                    seed = gr.Slider(label="Seed", minimum=0, maximum=MAX_SEED, step=1, value=0)
                # simplify mesh
                with gr.Row():
                    simplify_mesh = gr.Checkbox(label="Simplify mesh", value=False)
                    target_num_faces = gr.Slider(
                        label="Face number", minimum=10000, maximum=1000000, step=1000, value=100000
                    )
                # gen button
                button_gen = gr.Button("Generate")

        with gr.Column(scale=1):
            # glb file
            output_model = gr.Model3D(label="Geometry", height=512)

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
            fn=process_image,  # still need to click button_gen to get the 3d
            inputs=[input_image],
            outputs=[seg_image],
            cache_examples=False,
        )

    button_gen.click(process_image, inputs=[input_image], outputs=[seg_image]).then(
        get_random_seed, inputs=[randomize_seed, seed], outputs=[seed]
    ).then(
        process_3d,
        inputs=[seg_image, num_steps, cfg_scale, input_grid_res, seed, simplify_mesh, target_num_faces],
        outputs=[output_model],
    )

block.launch()