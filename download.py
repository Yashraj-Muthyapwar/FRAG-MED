
from pathlib import Path

from huggingface_hub import snapshot_download
from config import config


def main():
    model_id = config.EMBEDDING_MODEL_NAME
    local_dir: Path = config.EMBEDDING_MODEL_PATH

    # Ensure parent directory exists
    local_dir.parent.mkdir(parents=True, exist_ok=True)

    print(f"📥 Downloading '{model_id}' to '{local_dir}' ...")

    snapshot_download(
        repo_id=model_id,
        local_dir=str(local_dir),
        local_dir_use_symlinks=False,  # make real files instead of symlinks
    )

    print("\n✅ Download complete.")
    print(f"📂 Files saved under: {local_dir}")
    print("✅ You can now run your preprocessing + indexing with the LOCAL embedding model.")


if __name__ == "__main__":
    main()
