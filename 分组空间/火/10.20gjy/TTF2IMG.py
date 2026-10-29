"""将字体转为分类图片；路径以脚本目录为基准。"""
import argparse
from pathlib import Path
import time
from PIL import Image, ImageFont, ImageDraw
from fontTools.ttLib import TTFont

BASE_DIR = Path(__file__).resolve().parent


def char_to_img(all_chara, img_dir, uniMap, font, img_size, filename=None):
    count = 0
    for chara in all_chara:
        if ord(chara) not in uniMap:
            continue
        im = Image.new("RGB", (img_size, img_size), "white")
        draw = ImageDraw.Draw(im)
        # 按实际字形边界居中，避免新版 Pillow 的基线偏移造成裁切。
        left, top, right, bottom = draw.textbbox((0, 0), chara, font=font)
        draw.text(((img_size - right + left) / 2 - left,
                   (img_size - bottom + top) / 2 - top), chara, fill="black", font=font)
        folder = Path(img_dir) / chara
        folder.mkdir(parents=True, exist_ok=True)
        # 每个字体使用固定文件名，重复运行不会重复追加数据。
        name = filename or str(len(list(folder.glob("*.png"))))
        im.save(folder / (name + ".png"))
        count += 1
    return count


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--font-dir", type=Path, default=BASE_DIR / "ALL-Font")
    parser.add_argument("--output-dir", type=Path, default=BASE_DIR / "data")
    parser.add_argument("--limit", type=int, default=None, help="仅生成前 N 个汉字，用于快速检查")
    args = parser.parse_args()
    start = time.perf_counter()
    fonts = sorted(f for f in args.font_dir.iterdir() if f.suffix.lower() in {".ttf", ".otf"})
    if not fonts:
        raise ValueError("字体目录中没有 TTF/OTF 字体")
    chars = [chr(i) for i in range(19968, 26000)][:args.limit]
    total = 0
    for path in fonts:
        with TTFont(str(path)) as fontmap:
            cmap = fontmap.getBestCmap() or {}
        count = char_to_img(chars, args.output_dir, cmap,
                            ImageFont.truetype(str(path), 30), 30, path.stem)
        total += count
        print(f"{path.name}: {count} 张", flush=True)
    print(f"完成：{total} 张图片，耗时 {time.perf_counter()-start:.1f} 秒")


if __name__ == "__main__":
    main()
