"""Output presets; vertical dimensions, constant-quality H.264, unchanged FPS."""
from dataclasses import dataclass, asdict

@dataclass(frozen=True)
class Quality:
    name: str
    width: int
    height: int
    crf: int

    def report(self): return asdict(self)

PRESETS=(Quality('720p',720,1280,23),Quality('1080p',1080,1920,20),
         Quality('1440p',1440,2560,18),Quality('2160p',2160,3840,17))
QUALITY_IDS=tuple(p.name for p in PRESETS)
DEFAULT_QUALITY='1080p'

def get_quality(name=DEFAULT_QUALITY):
    for preset in PRESETS:
        if preset.name==name:return preset
    raise ValueError(f'未知画质预设：{name}')

def report_dimensions(report):
    # Archived renders were 720p; do not reinterpret them using today's default.
    config=report.get('output_quality')
    return (720,1280) if config is None else (get_quality(config['name']).width,get_quality(config['name']).height)
