# Mac 实验共享环境

`.venv` 位于本实验总文件夹，当前使用 Python 3.9.6，已安装实验 1 的依赖。其他实验可共用兼容的库，是否需要新环境须检查各实验依赖。

新开终端后执行：

```bash
cd "/Users/yio/Documents/研/数据科学与数字人文研究-丁家友陈必坤/DSDH-/文本处理系列实验"
source .venv/bin/activate
```

然后进入要做的实验文件夹，例如：

```bash
cd 实验1汉字处理/篆体字识别
```

编辑器解释器选择本目录的 `.venv/bin/python`。旧终端如果已激活原环境，请先 `deactivate`，再按上面的新路径激活。

依赖清单保留在实验 1 目录中的 `requirements-mac.txt` 和 `requirements-mac-lock.txt`。
