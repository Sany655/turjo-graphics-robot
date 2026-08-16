import gc
import os
import torch
import numpy as np
import cv2
from PIL import Image, ImageEnhance

def run_ryzen_local_pipeline(output_filename="final_car_4k_local.png"):
    # 1. Device Selection: DirectML for AMD Vega 7 iGPU or Multi-threaded CPU
    try:
        import torch_directml
        device = torch_directml.device()
        print("[*] Accelerated AMD DirectML Device detected (Vega 7 iGPU).")
    except ImportError:
        device = torch.device("cpu")
        torch.set_num_threads(6)  # Use all 6 physical cores of AMD Ryzen 5 5600G
        print("[*] DirectML not installed. Running on Multi-Threaded Ryzen 5 5600G CPU.")

    from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler

    PROMPT = (
        "masterpiece, RAW photo, ultra-realistic luxury sports car, sleek aerodynamic design, "
        "glossy metallic body with neon reflections, dramatic front three-quarter view, "
        "wet asphalt, modern city street at night, cinematic lighting, rim light, "
        "carbon fiber texture, shallow depth of field, photorealistic, 8k uhd"
    )

    NEGATIVE_PROMPT = (
        "low quality, blurry, deformed, cartoon, 3d render, plastic body, extra wheels, "
        "bad proportions, oversaturated, text, watermark"
    )

    # 2. Stage 1: Fast Photoreal Base Generation (912x512 16:9)
    print("\n[Stage 1/3] Loading Lightweight Photoreal Model (Realistic Vision V6.0)...")
    pipe = StableDiffusionPipeline.from_pretrained(
        "SG161222/Realistic_Vision_V6.0_B1_noVAE",
        torch_dtype=torch.float32,
        safety_checker=None
    ).to(device)

    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config, use_karras_sigmas=True)

    print("[*] Synthesizing 16:9 base composition (912x512)...")
    base_image = pipe(
        prompt=PROMPT,
        negative_prompt=NEGATIVE_PROMPT,
        width=912,
        height=512,
        num_inference_steps=25,
        guidance_scale=6.5
    ).images[0]

    base_image.save("stage1_ryzen_base.png")
    print("[✓] Base image saved: stage1_ryzen_base.png")

    # Free diffusion pipeline from RAM to prevent OOM
    del pipe
    gc.collect()

    # 3. Stage 2: 4K Super-Resolution
    print("\n[Stage 2/3] Upscaling to 4K UHD (3840x2160)...")
    img_np = np.array(base_image)
    target_w, target_h = 3840, 2160
    upscaled_4k = cv2.resize(img_np, (target_w, target_h), interpolation=cv2.INTER_LANCZOS4)

    # 4. Stage 3: Studio HDR Enhancement & Contrast Grading
    print("\n[Stage 3/3] Applying Color Grading & Edge Crispness...")
    pil_4k = Image.fromarray(upscaled_4k)
    sharpness = ImageEnhance.Sharpness(pil_4k).enhance(1.25)
    final_image = ImageEnhance.Contrast(sharpness).enhance(1.08)

    final_image.save(output_filename, quality=100)
    print("\n=======================================================")
    print(f"[SUCCESS] 4K Image Rendered locally on Ryzen 5600G!")
    print(f"Output saved: {output_filename} (3840 x 2160)")
    print("=======================================================")

if __name__ == "__main__":
    run_ryzen_local_pipeline()
