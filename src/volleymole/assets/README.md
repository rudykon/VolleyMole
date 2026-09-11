# 插画与标题字体

## Codex 内置生图素材

[飞行排球](illustrated/volley_ball.png)、[低位救球](illustrated/volley_save.png)、[网前组织](illustrated/volley_set.png) 为本项目使用 Codex 内置 `image_gen` 生成的排球主题插画。它们是装饰性运动插图，不代表原视频中任何特定球员或已经确认的比赛结果。

生成模式为内置工具，不使用 CLI/API 回退。使用统一的海军蓝、奶油白、橙黄、薄荷绿和珊瑚粉色系，采用手绘运动编辑插画风格。原始透明 PNG 保留 alpha 通道并复制到本目录，运行时不依赖 Codex 私有生成目录。

最终提示词全文及文件路径见 [prompts.json](illustrated/prompts.json)。视频标题单独排版，不要求生图模型生成数字名次或中文标题。

本次新增 [名次横幅](illustrated/rank_ribbon.png)、[飞扑救球](illustrated/volley_dive.png)、[低位接球](illustrated/volley_receive.png)、[跃起扣球](illustrated/volley_spike.png)，同样使用 Codex 内置生图、保留原始透明通道。新增素材的完整提示词见 [prompts-ranked.json](illustrated/prompts-ranked.json)。横幅中心由程序使用站酷快乐体叠加准确的“五佳球 · 第几球／十佳球 · 第几球”，确保所有名次正确、清晰，而非让模型猜文字。

## 五套可选装饰插画

`illustrated/themes/` 新增 `manga`（热血漫画）、`clay`（立体黏土）、`papercut`（层叠剪纸）、`ink`（东方水墨）、`retro`（复古丝网）五套素材，每套七张独立 PNG。它们均通过 Codex 内置 `image_gen` 逐张生成，没有使用 CLI/API 回退，也没有用精灵图切分代替独立素材。

源 PNG 原样复制，保留生成的 alpha 通道；渲染时仅由现有视频 UI 做等比缩放与合成。完整生成提示词、逐文件 SHA-256 和生成记录见 [themes/prompts.json](illustrated/themes/prompts.json)。运行时直接读取包内素材，不访问生成工具或私有生成路径。

插画仅作装饰，不代表真实球员、动作识别结论或比赛结果。它们与品牌标志、字体等素材的来源分开记录；项目代码的 MIT 许可不构成对 AI 生成图像独占版权或第三方权利的保证。风格用法和完整图集见 [插画风格指南](../../../docs/插画风格指南.md)。

## 五套动画转场材质

`transitions/` 包含 `velocity`、`paper`、`ink`、`prism`、`film` 五张独立不透明 PNG，分别采用竞技、纸艺、水墨、玻璃、胶片视觉语言。使用 Codex 内置 `image_gen` 逐张生成，原样复制到包内；提示词和 SHA-256 见 [生成记录](transitions/prompts.json)。没有使用 CLI/API 回退，也不在运行时依赖私有生成目录。

`transitions.py` 对原始材质做运行时缩放、遮罩与合成，生成入退场动画。可导出的透明 ProRes 4444 MOV，其 alpha 是程序的运动遮罩，不是生图原图透明通道。原 PNG 不被修改。用法见 [动画转场指南](../../../docs/动画转场指南.md)。

## 完整视觉套装补充素材

[aurora-ball.png](design_suites/aurora-ball.png) 为极光棱镜套装新增的透明玻璃排球，用 Codex 内置 `image_gen` 生成并原样保存，未使用 CLI/API 回退。完整提示词和 SHA-256 见 [design_suites/prompts.json](design_suites/prompts.json)。套装为玻璃图像匹配深蓝空间、银白文字和冷色转场，不再使用黏土人物作为主体。

其余完整套装复用原有插图与转场材质，代码负责字体、构图、配色和分层运动。所有源 PNG 和用户品牌标志保持不变；查看 [完整视觉套装指南](../../../docs/完整视觉套装指南.md)。

## 用户提供的品牌标志

[volleymole.svg](branding/volleymole.svg) 是用户提供的原始标志的逐字节包内副本，并非新生成的插画。[透明 PNG](branding/volleymole.png) 通过 `rasterize_logo.py` 使用 librsvg 按原始画布渲染，保留图形、颜色及宽高比，展示于每个转场顶部。`branding/volleymole.json` 记录 SVG、转换代码和 PNG 的 SHA-256；正常安装直接复用已校验的 PNG，不需要修改 site-packages。开发时若替换包内 SVG，需同步重新生成派生资源并重建 wheel。

## 中文标题字体

十套中英标题模板另使用包内 `NotoSansCJKsc-Bold.otf`、`NotoSerifCJKsc-Regular.otf`、`NotoSans-Bold.ttf`、`NotoSans-BoldItalic.ttf` 和 `NotoSerif-Regular.ttf`。前两者从已安装的 Noto CJK 字体集合提取 SC 字面，保留全部字形和字体元数据；后三者原样复制。字体本身均保留 SIL OFL 1.1，完整分发来源及许可见 [Noto CJK 声明](fonts/NOTICE-Noto-CJK.txt) 和 [Noto Core 声明](fonts/NOTICE-Noto-Core.txt)。声明中的 Debian 打包文件许可与字体本身的 OFL 条目分开记录；本项目没有复制 Debian 打包源码。模板用法见 [标题模板指南](../../../docs/标题模板指南.md)。

`fonts/ZCOOLKuaiLe-Regular.ttf` 是站酷快乐体，用于标题和转场大字。常规说明文字沿用运行参数中的正文字体。

字体与许可证下载自 [Google Fonts 的 ZCOOL KuaiLe 目录](https://github.com/google/fonts/tree/main/ofl/zcoolkuaile)。版权为 The ZCOOL KuaiLe Project Authors；随字体保留原始 [SIL OFL 1.1 许可证](https://raw.githubusercontent.com/google/fonts/main/ofl/zcoolkuaile/OFL.txt)，本地文件为 `fonts/OFL-ZCOOLKuaiLe.txt`。
