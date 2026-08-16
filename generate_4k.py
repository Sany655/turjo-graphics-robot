import gc
import os
import torch
import cv2
import numpy as np
from PIL import Image, ImageEnhance
from diffusers import AutoPipelineForText2Image, AutoPipelineForImage2Image, DPMSolverMultistepScheduler
from realesrgan import RealESRGANer
from basicsr.archs.rrdbnet_arch import RRDBNet

def run_low_resource_4k_pipeline(
    prompt: str,
    negative_prompt: str = None,
    output_filename: str = "final_car_4k.png"
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[*] Initializing Pipeline on device: {device}")

    if negative_prompt is None:
        negative_prompt = (
            "blurry, low quality, deformed, 3d render, cartoon, illustration, "
            "plastic reflections, duplicate wheels, oversaturated, watermark, bad proportions"
        )

    # -------------------------------------------------------------
    # STAGE 1: Base Latent Generation (SDXL RealVisXL @ 1280x720)
    # -------------------------------------------------------------
    print("\n[Stage 1/4] Loading SDXL Model (FP16 + VAE Tiling + CPU Offload)...")
    pipe = AutoPipelineForText2Image.from_pretrained(
        "SG161222/RealVisXL_V4.0",
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        variant="fp16" if device == "cuda" else None,
        use_safetensors=True
    )
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config, use_karras_sigmas=True)
    
    if device == "cuda":
        pipe.enable_model_cpu_offload()
        pipe.enable_vae_tiling()
    else:
        pipe.to(device)

    print("[*] Generating Base 16:9 Cinematic Render (1280x720)...")
    base_image = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        width=1280,
        height=720,
        guidance_scale=6.5,
        num_inference_steps=32
    ).images[0]
    base_image.save("stage1_base.png")
    print("[✓] Stage 1 Base image saved as stage1_base.png")

    # -------------------------------------------------------------
    # STAGE 2: Latent Micro-Detail Refiner (Low Denoise)
    # -------------------------------------------------------------
    print("\n[Stage 2/4] Refining Surface Textures & Reflections...")
    refiner_pipe = AutoPipelineForImage2Image.from_pipe(pipe)
    refined_image = refiner_pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        image=base_image,
        strength=0.28,
        guidance_scale=6.0,
        num_inference_steps=20
    ).images[0]
    refined_image.save("stage2_refined.png")
    print("[✓] Stage 2 Refined image saved as stage2_refined.png")

    # Free memory before running the Super-Resolution model
    del pipe
    del refiner_pipe
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()
    print("[*] VRAM flushed for Upscaling Stage.")

    # -------------------------------------------------------------
    # STAGE 3: Tiled Neural Super-Resolution (Upscale to 4K: 3840x2160)
    # -------------------------------------------------------------
    print("\n[Stage 3/4] Loading Real-ESRGAN x4plus (Tiled Processing)...")
    model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
    upscaler = RealESRGANer(
        scale=4,
        model_path="https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth",
        model=model,
        tile=512,        # 512px tile size prevents CUDA OOM on 8GB-15GB cards
        tile_pad=10,
        pre_pad=0,
        half=True if device == "cuda" else False,
        device=device
    )

    img_cv = cv2.cvtColor(np.array(refined_image), cv2.COLOR_RGB2BGR)
    # Outscale 3.0: 1280x720 * 3 = 3840x2160 (Exact 4K UHD)
    output_cv, _ = upscaler.enhance(img_cv, outscale=3.0)
    raw_4k_img = Image.fromarray(cv2.cvtColor(output_cv, cv2.COLOR_BGR2RGB))
    print("[✓] Stage 3 4K Upscaling complete.")

    # -------------------------------------------------------------
    # STAGE 4: Post-Processing & Color Grading
    # -------------------------------------------------------------
    print("\n[Stage 4/4] Applying Studio Color Grading & Contrast Boost...")
    sharpness = ImageEnhance.Sharpness(raw_4k_img).enhance(1.12)
    final_image = ImageEnhance.Contrast(sharpness).enhance(1.06)

    final_image.save(output_filename, format="PNG", quality=100)
    print(f"\n=======================================================")
    print(f"[SUCCESS] 4K Image Rendered: {output_filename}")
    print(f"Resolution: {final_image.size[0]} x {final_image.size[1]}")
    print(f"=======================================================")

if __name__ == "__main__":
    CAR_PROMPT = (
        "Ultra-realistic luxury sports car, sleek aerodynamic design, glossy metallic body, "
        "dramatic front three-quarter view, parked on modern city street at night, "
        "cinematic lighting, realistic reflections, wet asphalt, premium automotive photography, "
        "sharp details, carbon fiber textures, realistic materials, shallow depth of field, "
        "dramatic atmosphere, professional studio-quality composition, photorealistic, HDR, 8k uhd"
    )
    run_low_resource_4k_pipeline(prompt=CAR_PROMPT)
