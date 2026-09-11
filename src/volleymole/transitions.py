"""Generated material plates animated without changing rally or title timing."""
from pathlib import Path
from .common import APP

STYLE_LABELS = {'fade': '原版淡入淡出', 'velocity': '竞技斜切', 'paper': '纸艺翻页',
                'ink': '水墨流动', 'prism': '棱镜折光', 'film': '胶片光泄'}
STYLE_IDS = tuple(STYLE_LABELS)
ASSET_ROOT = APP / 'assets/transitions'


def validate_style(style):
    if style not in STYLE_IDS:
        raise ValueError(f'未知转场风格：{style}；可选 {", ".join(STYLE_IDS)}')
    return style


def transition_assets(style='fade'):
    validate_style(style)
    return [Path(__file__)] + ([] if style == 'fade' else [ASSET_ROOT / f'{style}.png'])


class TransitionRenderer:
    """BGR frames. Two 12-frame gestures surrounding an untouched title hold.

    A moving generated-material shutter covers the old image before revealing
    the next. The return gesture runs in the opposite direction. No random
    state, per-frame disk reads, source cropping, or changes to generated files.
    """

    def __init__(self, style, before, card, after, total=90, ramp=12):
        import cv2
        import numpy as np
        validate_style(style)
        if total < 2 * ramp or ramp < 2:
            raise ValueError('转场帧数不足')
        if before.shape != card.shape or after.shape != card.shape or card.ndim != 3 or card.shape[2] != 3:
            raise ValueError('转场输入必须为同尺寸 BGR 三通道帧')
        self.style, self.before, self.card, self.after = style, before, card, after
        self.total, self.ramp = total, ramp
        if style == 'fade':
            return
        raw = cv2.imread(str(transition_assets(style)[1]), cv2.IMREAD_COLOR)
        if raw is None:
            raise ValueError(f'缺失内置转场素材：{style}')
        h, w = card.shape[:2]
        # Cover fit keeps source aspect ratio; the original PNG is untouched.
        scale = max(w / raw.shape[1], h / raw.shape[0])
        raw = cv2.resize(raw, (round(raw.shape[1] * scale), round(raw.shape[0] * scale)), interpolation=cv2.INTER_LANCZOS4)
        y0, x0 = (raw.shape[0] - h) // 2, (raw.shape[1] - w) // 2
        self.plate = raw[y0:y0+h, x0:x0+w].astype(np.float32)
        y, x = np.mgrid[0:h, 0:w].astype(np.float32)
        x /= max(1, w-1)
        y /= max(1, h-1)
        if style == 'velocity':
            field = .72*x + .28*y
        elif style == 'paper':
            field = .60*x + .40*y + .075*np.sin(y*5) + .008*np.sin(y*91)
        elif style == 'ink':
            luma = cv2.cvtColor(self.plate, cv2.COLOR_BGR2GRAY) / 255
            field = .62*y + .20*x + .13*np.sin(x*9+y*5) + .10*luma
        elif style == 'prism':
            # Offset optical shutters produce a staggered, faceted opening.
            field = .68*x + .19*y + .13*np.floor(y*4)/4
        else:
            field = np.sqrt(((x-.18)*.72)**2 + ((y-.55)*1.05)**2)
        self.field = ((field-field.min()) / max(float(np.ptp(field)), 1e-6))[:, :, None]

    def masks(self, progress, reverse=False):
        """Leading/trailing edges; the midpoint is fully covered for clean cuts."""
        import numpy as np
        if self.style == 'fade':
            raise ValueError('原版淡入淡出不提供材质遮罩')
        progress = float(np.clip(progress, 0, 1))
        p = progress*progress*(3-2*progress)
        field = 1-self.field if reverse else self.field
        softness = {'velocity': .025, 'paper': .018, 'ink': .15, 'prism': .06, 'film': .55}[self.style]
        # A full material cover between two moving edges, not a static fade.
        width = 1+2*softness+.35
        travel = p*(1+width+2*softness)-softness
        def mask(edge):
            a = np.clip((edge-field)/softness, 0, 1)
            return a*a*(3-2*a)
        return mask(travel), mask(travel-width)

    def material_layer(self, progress, reverse=False):
        """Straight-alpha BGRA overlay for an external editor; cut at midpoint."""
        import numpy as np
        lead, trail = self.masks(progress, reverse)
        return np.concatenate((self.plate.astype(np.uint8), np.rint((lead-trail)*255).astype(np.uint8)), axis=2)

    def _gesture(self, old, new, progress, reverse=False):
        import numpy as np
        if progress <= 0:
            return old.copy()
        if progress >= 1:
            return new.copy()
        lead, trail = self.masks(progress, reverse)
        mixed = old.astype(np.float32)*(1-lead) + self.plate*(lead-trail) + new.astype(np.float32)*trail
        return np.clip(np.rint(mixed), 0, 255).astype(np.uint8)

    def frame(self, index):
        import cv2
        if not 0 <= index < self.total:
            raise ValueError('转场帧索引越界')
        if self.style == 'fade':
            p = max(0., min(1., index/(self.ramp-1), (self.total-1-index)/(self.ramp-1)))
            alpha = p*p*(3-2*p)
            base = self.before if index < self.total//2 else self.after
            return cv2.addWeighted(self.card, alpha, base, 1-alpha, 0)
        if index < self.ramp:
            return self._gesture(self.before, self.card, index/(self.ramp-1))
        if index >= self.total-self.ramp:
            return self._gesture(self.card, self.after, (index-(self.total-self.ramp))/(self.ramp-1), reverse=True)
        return self.card.copy()
