# 本文件用于下载并获取 embedding 向量化模型

from pathlib import Path
from langchain_huggingface import HuggingFaceEmbeddings
from huggingface_hub import snapshot_download


def get_embeddings(model_name="BAAI/bge-small-zh-v1.5", device="cpu", **kwargs):
    """获取 embedding 模型实例；首次调用时自动下载到本地 models/ 目录。

    后续调用会直接使用本地文件（local_files_only=True），可离线使用。
    """
    local_dir = Path("models") / model_name.replace("/", "_")
    if not local_dir.exists():
        print(f'⚠️  首次使用嵌入模型，正在下载到 {local_dir.absolute()}')
        print("💡 提示：需要联网（需可访问 Hugging Face），完成后可离线使用")
        snapshot_download(repo_id=model_name, local_dir=local_dir)
        print('✅ 下载完成！')

    model_kwargs = {
        "device": device,
        "local_files_only": True,  # 仅使用本地文件，避免重复下载
    }

    return HuggingFaceEmbeddings(
        model_name=str(local_dir),  # 使用本地已下载的模型
        model_kwargs=model_kwargs,
        **kwargs,
    )
