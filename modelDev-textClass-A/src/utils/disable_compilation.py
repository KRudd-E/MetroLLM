""" disable_compilation.py
Disable PyTorch compilation features to avoid CUDA compilation issues in HPC environments.
Must be done early in the script, ideally before importing torch or transformers.
"""
def disable_compilation():
    import os

    # Disable TorchDynamo and compilation
    os.environ["TORCHDYNAMO_DISABLE"] = "1"
    os.environ["TORCH_COMPILE_DEBUG"] = "0"
    os.environ["TORCH_LOGS"] = ""

    # Disable Triton
    os.environ["TRITON_DISABLE_LINE_INFO"] = "1"

    # Set attention implementation to eager (non-compiled)
    os.environ["ATTENTION_IMPLEMENTATION"] = "eager"
    
    # Disable flash attention specifically for ModernBERT
    os.environ["FLASH_ATTENTION_FORCE_EAGER"] = "1"
    os.environ["MODERNBERT_ATTENTION_IMPLEMENTATION"] = "eager"
