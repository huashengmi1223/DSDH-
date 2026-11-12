"""使用随附的 30 篇白话杂文重做词项对比和图表。"""
import argparse
from collections import Counter
from importlib.metadata import version
import json
from math import isclose, log, sqrt
from pathlib import Path
import re
from tempfile import TemporaryDirectory
import unicodedata

import jieba
import matplotlib
matplotlib.use("Agg")
from matplotlib import font_manager, pyplot as plt, ticker
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from 词云 import draw_cloud

ROOT = Path(__file__).resolve().parent
AUTHORS = ("鲁迅", "周作人")
COLORS = {"鲁迅":"#235B80", "周作人":"#AB522B"}
HAN = re.compile(r"[\u3400-\u9fff\U00020000-\U0002ffff]+")
CHINESE_NUMERALS = re.compile(r"[〇零一二三四五六七八九十百千万亿两廿卅]+")


def read(name):
    return json.loads((ROOT / name).read_text(encoding="utf-8"))


def token_documents(poems, rules, cache):
    jieba.setLogLevel(jieba.logging.WARN)
    cutter = jieba.Tokenizer()
    cutter.tmp_dir = cache
    cutter.initialize()
    for term in rules["专名与时代用语"]:
        cutter.add_word(term, freq=10_000)
    stop = set(rules["停用词"].split())
    for article in poems:
        paragraphs = []
        for paragraph in article["body"]:
            norm = unicodedata.normalize("NFKC", paragraph)
            groups = HAN.findall(re.sub(r"(?<=[\u3400-\u9fff]) +(?=[\u3400-\u9fff])", "", norm))
            tokens = [t for g in groups for t in cutter.cut(g, HMM=False)
                      if len(t)>=2 and t not in stop and not CHINESE_NUMERALS.fullmatch(t)]
            paragraphs.append(tokens)
        yield {**article, "tokens":paragraphs}


def fit(documents):
    model = TfidfVectorizer(
        tokenizer=str.split, token_pattern=None, lowercase=False, ngram_range=(1,1),
        min_df=2, max_df=.90, norm="l2", smooth_idf=True, sublinear_tf=False,
    )
    matrix = model.fit_transform(" ".join(t for p in d["tokens"] for t in p) for d in documents)
    terms = model.get_feature_names_out()
    result = {"矩阵形状":list(matrix.shape), "按作者排序":{}}
    for author in AUTHORS:
        ids = [i for i,d in enumerate(documents) if d["author"]==author]
        rows = []
        for k in np.flatnonzero(np.asarray(matrix[ids].sum(axis=0)).ravel()):
            values = matrix[ids,k].toarray().ravel()
            hits = [documents[i]["id"] for i,v in zip(ids,values) if v]
            row = {"词":str(terms[k]),"均值":float(values.mean()),"篇数":len(hits),"命中":hits,
                   "最高单篇占比":float(values.max()/values.sum()),
                   "最高单篇编号":documents[ids[int(values.argmax())]]["id"]}
            for year in (1925,1926):
                which = [j for j,i in enumerate(ids) if documents[i]["year"]==year]
                row[str(year)] = float(values[which].mean())
            rows.append(row)
        result["按作者排序"][author] = sorted(rows,key=lambda r:(-r["均值"],-r["篇数"],r["词"]))
    return result


def verify(documents, result):
    """独立用计数和公式核对词表、分值和篇数。"""
    counters = [Counter(t for p in d["tokens"] for t in p) for d in documents]
    n = len(counters)
    dfs = Counter(w for c in counters for w in c)
    idfs = {w: 1+log((1+n)/(1+df)) for w,df in dfs.items() if 2<=df<=.9*n}
    assert result["矩阵形状"]==[n,len(idfs)]
    vectors = []
    for c in counters:
        vals = {w:n*idfs[w] for w,n in c.items() if w in idfs}
        denom = sqrt(sum(v*v for v in vals.values()))
        vectors.append({w:v/denom for w,v in vals.items()} if denom else {})
    for author, rows in result["按作者排序"].items():
        subset = [v for d,v in zip(documents,vectors) if d["author"]==author]
        coverage = Counter(w for v in subset for w in v)
        assert set(coverage)=={r["词"] for r in rows}
        for r in rows:
            term = r["词"]
            assert r["篇数"]==coverage[term]
            values = [v.get(term,0.) for v in subset]
            assert isclose(r["均值"],sum(values)/len(subset),abs_tol=1e-12)
            assert isclose(r["最高单篇占比"],max(values)/sum(values),abs_tol=1e-12)
            for year in (1925,1926):
                yearly = [v.get(term,0.) for d,v in zip(documents,vectors)
                          if d["author"]==author and d["year"]==year]
                assert isclose(r[str(year)],sum(yearly)/len(yearly),abs_tol=1e-12)


def cjk_font(explicit):
    if explicit:
        return font_manager.FontProperties(fname=str(Path(explicit).resolve()))
    for family in ("Hiragino Sans GB", "PingFang SC", "Noto Sans CJK SC",
                   "Noto Sans SC", "Microsoft YaHei", "SimHei", "WenQuanYi Zen Hei"):
        try:
            name = font_manager.findfont(font_manager.FontProperties(family=family),
                                         fallback_to_default=False)
            return font_manager.FontProperties(fname=name)
        except ValueError:
            pass
    raise SystemExit("请用 --font 指定支持简体中文的字体文件。")


def bar_plot(result, font, path):
    rank = result["按作者排序"]
    values = {a:{r["词"]:r["均值"] for r in rank[a]} for a in AUTHORS}
    terms = {r["词"] for a in AUTHORS for r in rank[a][:8]}
    terms = sorted(terms,key=lambda t:(-max(values[a].get(t,0) for a in AUTHORS),t))
    n = len(terms)
    coords = np.arange(n)
    fig, ax = plt.subplots(figsize=(8.8, 5.4))
    fig.subplots_adjust(left=.19, right=.94, top=.87, bottom=.15)
    for j, author in enumerate(AUTHORS):
        scores = [values[author].get(t, 0.) for t in terms]
        ax.barh(coords + (j-.5)*.34, scores, .29, label=author, color=COLORS[author], zorder=3)
    ax.set_yticks(coords, terms, fontproperties=font, fontsize=12)
    ax.set_ylim(n-.5, -.5)
    ax.tick_params(axis="both", length=0, pad=7, labelsize=11)
    high = max(values[a].get(t,0.) for a in AUTHORS for t in terms)
    unit = .01 if high <= .10 else .02
    ax.set_xlim(0, np.ceil(high/unit)*unit)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(unit))
    ax.xaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))
    ax.set_xlabel("作者平均 TF-IDF", fontproperties=font, fontsize=12, labelpad=9)
    ax.grid(axis="x", color="#D7DEE2", linewidth=.7, zorder=0)
    ax.spines[["top","right","bottom","left"]].set_visible(False)
    ax.legend(loc="lower left", bbox_to_anchor=(-.015, 1.02), ncols=2, frameon=False, prop=font)
    fig.savefig(path, dpi=260, facecolor="white")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--font", help="支持中文的字体文件路径")
    parser.add_argument("--out", default=ROOT/"复跑结果", type=Path)
    parser.add_argument("--figures", type=Path)
    args = parser.parse_args()
    figures = args.figures or args.out
    for directory in (figures,args.out):
        directory.mkdir(exist_ok=True,parents=True)
    poems = read("语料.json")
    assert len({d["id"] for d in poems})==30
    assert Counter((d["author"],d["year"]) for d in poems)=={
        ("鲁迅",1925):8,("鲁迅",1926):7,("周作人",1925):8,("周作人",1926):7}
    assert all(d["chars"]==sum(len(group) for p in d["body"] for group in HAN.findall(p))
               for d in poems)
    rules = read("分词规则.json")
    with TemporaryDirectory() as cache:
        documents = list(token_documents(poems,rules,cache))
    result = fit(documents)
    verify(documents,result)
    result["公式复算"] = "词表维度、所有非零作者均值及命中篇数一致"
    result["版本"] = {pkg: version(pkg) for pkg in ("jieba","numpy","scikit-learn","matplotlib","Pillow")}
    result["逐篇分词"] = {d["id"]:d["tokens"] for d in documents}
    (args.out/"结果.json").write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    font = cjk_font(args.font)
    bar_plot(result,font,figures/"关键词条形对照图.png")
    draw_cloud(result,font.get_file(),figures/"关键词词云对照图.png")
    print("独立复算通过。",result["矩阵形状"])
    for author, rows in result["按作者排序"].items():
        print(author, "；".join(f'{r["词"]} {r["均值"]:.4f}（{r["篇数"]}篇）' for r in rows[:12]))


if __name__=="__main__":
    main()
