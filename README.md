# Fooocus
This is my first Git Repository 
Collecting pygit2==1.15.1
  Downloading pygit2-1.15.1-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (3.3 kB)
Requirement already satisfied: cffi>=1.16.0 in /usr/local/lib/python3.10/dist-packages (from pygit2==1.15.1) (1.17.1)
Requirement already satisfied: pycparser in /usr/local/lib/python3.10/dist-packages (from cffi>=1.16.0->pygit2==1.15.1) (2.22)
Downloading pygit2-1.15.1-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (5.1 MB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 5.1/5.1 MB 48.1 MB/s eta 0:00:00
Installing collected packages: pygit2
Successfully installed pygit2-1.15.1
/content
Cloning into 'Fooocus'...
remote: Enumerating objects: 6718, done.
remote: Counting objects: 100% (31/31), done.
remote: Compressing objects: 100% (21/21), done.
remote: Total 6718 (delta 11), reused 22 (delta 8), pack-reused 6687 (from 1)
Receiving objects: 100% (6718/6718), 33.26 MiB | 19.91 MiB/s, done.
Resolving deltas: 100% (3870/3870), done.
/content/Fooocus
Already up-to-date
Update succeeded.
[System ARGV] ['entry_with_update.py', '--share', '--always-high-vram']
Python 3.10.12 (main, Sep 11 2024, 15:47:36) [GCC 11.4.0]
Fooocus version: 2.5.5
Error checking version for torchsde: No package metadata was found for torchsde
Installing requirements
[Cleanup] Attempting to delete content of temp dir /tmp/fooocus
[Cleanup] Cleanup successful
Downloading: "https://huggingface.co/lllyasviel/misc/resolve/main/xlvaeapp.pth" to /content/Fooocus/models/vae_approx/xlvaeapp.pth

100% 209k/209k [00:00<00:00, 7.56MB/s]
Downloading: "https://huggingface.co/lllyasviel/misc/resolve/main/vaeapp_sd15.pt" to /content/Fooocus/models/vae_approx/vaeapp_sd15.pth

100% 209k/209k [00:00<00:00, 7.17MB/s]
Downloading: "https://huggingface.co/mashb1t/misc/resolve/main/xl-to-v1_interposer-v4.0.safetensors" to /content/Fooocus/models/vae_approx/xl-to-v1_interposer-v4.0.safetensors

100% 5.40M/5.40M [00:00<00:00, 93.2MB/s]
Downloading: "https://huggingface.co/lllyasviel/misc/resolve/main/fooocus_expansion.bin" to /content/Fooocus/models/prompt_expansion/fooocus_expansion/pytorch_model.bin

100% 335M/335M [00:01<00:00, 328MB/s]
Downloading: "https://huggingface.co/lllyasviel/fav_models/resolve/main/fav/juggernautXL_v8Rundiffusion.safetensors" to /content/Fooocus/models/checkpoints/juggernautXL_v8Rundiffusion.safetensors

100% 6.62G/6.62G [00:37<00:00, 191MB/s]
Downloading: "https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/resolve/main/sd_xl_offset_example-lora_1.0.safetensors" to /content/Fooocus/models/loras/sd_xl_offset_example-lora_1.0.safetensors

100% 47.3M/47.3M [00:00<00:00, 245MB/s]
Total VRAM 15102 MB, total RAM 12979 MB
Set vram state to: HIGH_VRAM
Always offload VRAM
Device: cuda:0 Tesla T4 : native
VAE dtype: torch.float32
Using pytorch cross attention
Refiner unloaded.
IMPORTANT: You are using gradio version 3.41.2, however version 4.29.0 is available, please upgrade.
--------
Running on local URL:  http://127.0.0.1:7865
Running on public URL: https://9c43b294d3c94ea967.gradio.live

This share link expires in 72 hours. For free permanent hosting and GPU upgrades, run `gradio deploy` from Terminal to deploy to Spaces (https://huggingface.co/spaces)
model_type EPS
UNet ADM Dimension 2816
Using pytorch attention in VAE
Working with z of shape (1, 4, 32, 32) = 4096 dimensions.
Using pytorch attention in VAE
extra {'cond_stage_model.clip_l.text_projection', 'cond_stage_model.clip_l.logit_scale'}
left over keys: dict_keys(['cond_stage_model.clip_l.transformer.text_model.embeddings.position_ids'])
loaded straight to GPU
Requested to load SDXL
Loading 1 new model
Base model loaded: /content/Fooocus/models/checkpoints/juggernautXL_v8Rundiffusion.safetensors
VAE loaded: None
Request to load LoRAs [('sd_xl_offset_example-lora_1.0.safetensors', 0.1)] for model [/content/Fooocus/models/checkpoints/juggernautXL_v8Rundiffusion.safetensors].
Loaded LoRA [/content/Fooocus/models/loras/sd_xl_offset_example-lora_1.0.safetensors] for UNet [/content/Fooocus/models/checkpoints/juggernautXL_v8Rundiffusion.safetensors] with 788 keys at weight 0.1.
Fooocus V2 Expansion: Vocab with 642 words.
Fooocus Expansion engine loaded for cuda:0, use_fp16 = True.
Requested to load SDXLClipModel
Requested to load GPT2LMHeadModel
Loading 2 new models
[Fooocus Model Management] Moving model(s) has taken 0.61 seconds
2024-09-27 15:37:04.648095: E external/local_xla/xla/stream_executor/cuda/cuda_fft.cc:485] Unable to register cuFFT factory: Attempting to register factory for plugin cuFFT when one has already been registered
2024-09-27 15:37:04.919406: E external/local_xla/xla/stream_executor/cuda/cuda_dnn.cc:8454] Unable to register cuDNN factory: Attempting to register factory for plugin cuDNN when one has already been registered
2024-09-27 15:37:04.995561: E external/local_xla/xla/stream_executor/cuda/cuda_blas.cc:1452] Unable to register cuBLAS factory: Attempting to register factory for plugin cuBLAS when one has already been registered
2024-09-27 15:37:05.414055: I tensorflow/core/platform/cpu_feature_guard.cc:210] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.
To enable the following instructions: AVX2 AVX512F FMA, in other operations, rebuild TensorFlow with the appropriate compiler flags.
2024-09-27 15:37:08.254455: W tensorflow/compiler/tf2tensorrt/utils/py_utils.cc:38] TF-TRT Warning: Could not find TensorRT
Started worker with PID 5124
App started successful. Use the app with http://127.0.0.1:7865/ or 127.0.0.1:7865 or https://9c43b294d3c94ea967.gradio.live
