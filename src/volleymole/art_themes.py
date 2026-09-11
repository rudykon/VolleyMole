"""Built-in illustration palettes. Selection is explicit, never process-global."""
from dataclasses import dataclass
from pathlib import Path
from .common import APP


@dataclass(frozen=True)
class ArtTheme:
    name: str
    label: str
    ink: tuple
    cream: tuple
    colors: tuple

    @property
    def root(self) -> Path:
        base = APP / 'assets/illustrated'
        return base if self.name == 'default' else base / 'themes' / self.name


_THEMES = (
    ArtTheme('default', '原版手绘', (13,22,40), (255,246,223),
             ((255,198,70), (85,231,199), (255,115,142))),
    ArtTheme('manga', '热血漫画', (28,25,26), (255,247,228),
             ((245,179,48), (238,94,69), (255,211,101))),
    ArtTheme('clay', '立体黏土', (51,36,64), (255,248,240),
             ((166,223,196), (197,171,227), (248,183,157))),
    ArtTheme('papercut', '层叠剪纸', (22,68,72), (255,247,226),
             ((235,132,106), (227,183,75), (135,200,185))),
    ArtTheme('ink', '东方水墨', (34,40,39), (248,246,235),
             ((201,91,70), (139,181,158), (197,181,145))),
    ArtTheme('retro', '复古丝网', (24,63,69), (251,241,217),
             ((218,127,65), (215,176,80), (134,179,168))),
)
THEME_IDS = tuple(theme.name for theme in _THEMES)


def get_theme(name='default') -> ArtTheme:
    for theme in _THEMES:
        if theme.name == name:
            return theme
    raise ValueError(f'未知插画风格：{name}；可选：{", ".join(THEME_IDS)}')
