import os
import sys
from huggingface_hub import hf_hub_download, login

def main():
    print("=" * 60)
    print("SAM 3 Checkpoint Downloader")
    print("=" * 60)

    # Read HF_TOKEN from environment if present
    hf_token = os.getenv("HF_TOKEN")
    if hf_token:
        print("Using Hugging Face token from environment...")
        login(token=hf_token)

    # Output directory relative to sam3 repo
    output_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "checkpoints"))
    os.makedirs(output_dir, exist_ok=True)

    print(f"Target download directory: {output_dir}")
    print("Downloading SAM 3 checkpoint (sam3.pt) from Hugging Face...")

    try:
        file_path = hf_hub_download(
            repo_id="facebook/sam3",
            filename="sam3.pt",
            local_dir=output_dir,
        )
        print("=" * 60)
        print(f"SUCCESS: SAM 3 checkpoint downloaded to:\n{file_path}")
        print("=" * 60)
    except Exception as e:
        print("\nERROR downloading checkpoint:")
        print(e)
        print("\nPlease ensure you have:")
        print("1. Accepted terms at: https://huggingface.co/facebook/sam3")
        print("2. Run 'hf auth login' or set HF_TOKEN environment variable.")
        sys.exit(1)

if __name__ == "__main__":
    main()
