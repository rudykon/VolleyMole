# 插画与标题字体

## Codex 内置生图素材

[飞行排球](illustrated/volley_ball.png)、[低位救球](illustrated/volley_save.png)、[网前组织](illustrated/volley_set.png) 为本项目使用 Codex 内置 `image_gen` 生成的排球主题插画。它们是装饰性运动插图，不代表原视频中任何特定球员或已经确认的比赛结果。

生成模式为内置工具，不使用 CLI/API 回退。使用统一的海军蓝、奶油白、橙黄、薄荷绿和珊瑚粉色系，采用手绘运动编辑插画风格。原始透明 PNG 保留 alpha 通道并复制到本目录，运行时不依赖 Codex 私有生成目录。

最终提示词全文及文件路径见 [prompts.json](illustrated/prompts.json)。视频标题单独排版，不要求生图模型生成数字名次或中文标题。

本次新增 [名次横幅](illustrated/rank_ribbon.png)、[飞扑救球](illustrated/volley_dive.png)、[低位接球](illustrated/volley_receive.png)、[跃起扣球](illustrated/volley_spike.png)，同样使用 Codex 内置生图、保留原始透明通道。新增素材的完整提示词见 [prompts-ranked.json](illustrated/prompts-ranked.json)。横幅中心由程序使用站酷快乐体叠加准确的“五佳球 · 第几球／十佳球 · 第几球”，确保所有名次正确、清晰，而非让模型猜文字。

## 用户提供的品牌标志

[volleymole.svg](../../../volleymole.svg) 是用户提供的原始标志，并非新生成的插画。[透明 PNG](branding/volleymole.png) 通过 `rasterize_logo.py` 使用 librsvg 按原始画布渲染，保留图形、颜色及宽高比，展示于每个转场顶部。`branding/volleymole.json` 记录 SVG、转换代码和 PNG 的 SHA-256，更换源标志会自动更新派生资源。

## 中文标题字体

`fonts/ZCOOLKuaiLe-Regular.ttf` 是站酷快乐体，用于标题和转场大字。常规说明文字沿用运行参数中的正文字体。

字体与许可证下载自 [Google Fonts 的 ZCOOL KuaiLe 目录](https://github.com/google/fonts/tree/main/ofl/zcoolkuaile)。版权为 The ZCOOL KuaiLe Project Authors；随字体保留原始 [SIL OFL 1.1 许可证](https://raw.githubusercontent.com/google/fonts/main/ofl/zcoolkuaile/OFL.txt)，本地文件为 `fonts/OFL-ZCOOLKuaiLe.txt`。
