# 统一模板库

所有模板固定保存在这个目录。命令行与网页的读取、导入和保存遵循同一套规则。

```text
templates/
├── builtin/     五套内置模板
├── custom/      个人模板、网页保存的模板
└── bundles/     已确认的完整定稿包与校验记录
```

## 内置模板

| 名称 | 配置 |
| --- | --- |
| 赤线竞技 | [matchday.json](builtin/matchday.json) |
| 纸上球场 | [atelier.json](builtin/atelier.json) |
| 水墨 · 墨间回合 | [sumi.json](builtin/sumi.json) |
| 极光棱镜 | [aurora.json](builtin/aurora.json) |
| 胶片纪事 | [archive.json](builtin/archive.json) |

内置配置以 `builtin/` 为唯一源码位置。`custom/` 存放导入与网页保存的个人 JSON，通过名称选择即可使用。

## 已确认的定稿包

水墨 v2.0.0（2026-09-30）保存在 `bundles/sumi-v2.0.0/`，同目录的 `.zip` 为完整备份，`.sha256`、`.validation.json`、`.tests.log` 为校验与验证记录。包内 `preview.png` 是确认版预览；详细用法见[水墨模板定稿](../docs/模板与配音.md#sumi)。

```bash
# 在项目根目录校验
.venv/bin/python templates/bundles/sumi-v2.0.0/sumi.py --verify
```

内置配置和本文随源码保存，个人模板及定稿包保留在本机。完整目录规则、独立安装与迁移说明见[模板目录](../docs/模板与配音.md#template-library)。
