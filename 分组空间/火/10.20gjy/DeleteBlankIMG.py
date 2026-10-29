"""删除全白图片；忽略隐藏文件，支持任意图片尺寸。"""
import argparse
from pathlib import Path
import numpy as np
from PIL import Image
from tqdm import tqdm

BASE_DIR = Path(__file__).resolve().parent


def clean_blank(img_dir=BASE_DIR / "data"):
    img_dir = Path(img_dir)
    if not img_dir.is_dir():
        raise FileNotFoundError(f"未找到 {img_dir}，请先运行 TTF2IMG.py")
    removed = 0
    folders = sorted(p for p in img_dir.iterdir() if p.is_dir() and not p.name.startswith("."))
    for folder in tqdm(folders, desc="正在清除空白图片"):
        for path in folder.iterdir():
            if path.suffix.lower() not in {".png", ".jpg", ".jpeg", ".bmp"}:
                continue
            with Image.open(path) as image:
                blank = np.all(np.asarray(image.convert("RGB")) == 255)
            if blank:
                path.unlink()
                removed += 1
        if not any(folder.iterdir()):
            folder.rmdir()
    print(f"清理完成：删除 {removed} 张全白图片")
    return removed


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=BASE_DIR / "data")
    clean_blank(parser.parse_args().data_dir)
