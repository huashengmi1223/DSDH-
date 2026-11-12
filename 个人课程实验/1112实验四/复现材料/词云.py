"""把作者平均 TF-IDF 画成并排的彩色词云。"""
from functools import lru_cache
from math import cos, pi, sin, sqrt
from pathlib import Path
from random import Random

from PIL import Image, ImageDraw, ImageFont

TOP_WORDS = 42
SONGTI = Path("/System/Library/Fonts/Supplemental/Songti.ttc")
COLORS = {
    "鲁迅": ("#B43C35", "#204F55", "#B17323", "#4D5F83", "#A05762", "#477B66", "#B78026"),
    "周作人": ("#245D62", "#B84E3C", "#AC7720", "#405C83", "#A15767", "#467C64", "#B87342"),
}


def draw_cloud(result, fallback_font, path):
    # Use the regular and bold faces of Songti SC where available.
    font_path = str(SONGTI) if SONGTI.exists() else fallback_font
    regular, bold = (6, 1) if SONGTI.exists() else (0, 0)

    @lru_cache(maxsize=128)
    def font(size, strong=False):
        return ImageFont.truetype(font_path, size, index=bold if strong else regular)

    def word_stamp(word, size, ink, vertical, strong):
        glyphs = list(word) if vertical else [word]
        face = font(size, strong)
        boxes = [face.getbbox(glyph) for glyph in glyphs]
        widths = [right-left for left, _, right, _ in boxes]
        if vertical:
            step = round(size*1.03)
            height = (len(glyphs)-1)*step + max(b-t for _,t,_,b in boxes)
        else:
            step = 0
            height = boxes[0][3]-boxes[0][1]
        width = max(widths)
        stamp = Image.new("RGBA", (width, height))
        brush = ImageDraw.Draw(stamp)
        for i, (glyph, (left,top,_,_), w) in enumerate(zip(glyphs,boxes,widths)):
            brush.text(((width-w)//2-left,i*step-top),glyph,font=face,fill=ink)
        return stamp

    def pack(rows, palette, seed, scale):
        width, height = 1080, 680
        center_x, center_y = width//2, height//2
        laid_out = []
        rng = Random(seed)
        hi, lo = rows[0]["均值"], rows[-1]["均值"]
        for rank, row in enumerate(rows):
            ratio = (row["均值"]-lo)/(hi-lo) if hi>lo else 1
            order = (1-rank/(len(rows)-1))**2
            size = round((31 + 131*(.65*ratio**.68+.35*order))*scale)
            word = row["词"]
            color = palette[rank] if rank < 4 else rng.choice(palette)
            angle = rng.random()*2*pi
            vertical = rank in {5, 12, 19, 26, 33, 39} and 2<=len(word)<=4
            options = (vertical, False) if vertical else (False,)
            for is_vertical in options:
                stamp = word_stamp(word,size,color,is_vertical,rank<4)
                w,h = stamp.size
                # Try points from the center out, using a golden-angle spiral.
                for n in range(2200):
                    radius = sqrt(n/2199)
                    theta = angle+n*2.39996323
                    x = round(center_x + cos(theta)*radius*530 - w/2)
                    y = round(center_y + sin(theta)*radius*330 - h/2)
                    if x<8 or x+w>width-8 or y<8 or y+h>height-8:
                        continue
                    gap = 11
                    if any(x < right+gap and x+w+gap > left and
                           y < bottom+gap and y+h+gap > top
                           for left,top,right,bottom,_ in laid_out):
                        continue
                    laid_out.append((x,y,x+w,y+h,stamp))
                    break
                else:
                    continue
                break
            else:
                return None

        panel = Image.new("RGBA",(width,height))
        for x,y,_,_,stamp in laid_out:
            panel.alpha_composite(stamp,(x,y))
        return panel.crop(panel.getbbox())

    # Both sides use the same maximum scale and typography.
    ranking = result["按作者排序"]
    for scale in (1.0, .96, .92, .88, .84, .80, .76, .72):
        clouds = [pack(ranking[author][:TOP_WORDS], palette, 21+i*71, scale)
                  for i,(author,palette) in enumerate(COLORS.items())]
        if all(cloud is not None for cloud in clouds):
            break
    else:
        raise ValueError("词云空间不足，无法排入全部词项。")

    canvas = Image.new("RGB",(2340,923),"white")
    brush = ImageDraw.Draw(canvas)
    centers = (585,1755)
    for center, (author,palette), cloud in zip(centers,COLORS.items(),clouds):
        brush.text((center,78),author,anchor="mm",font=font(55,True),fill="#263138")
        brush.rounded_rectangle((center-22,133,center+22,138),radius=2,fill=palette[0])
        x = center-cloud.width//2
        y = 168+(675-cloud.height)//2
        canvas.paste(cloud,(x,y),cloud)
    brush.line((1170,200,1170,808),fill="#E5E3DF",width=2)
    canvas.save(path,dpi=(260,260))
