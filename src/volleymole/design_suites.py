"""Five art-directed identities: typography, illustration, palette and motion.

Selection is immutable and explicit. Legacy mix-and-match designs are retained.
"""
from dataclasses import dataclass, asdict
from pathlib import Path
import math
from .common import APP
from .art_themes import ArtTheme, get_theme
from .title_templates import FONTS, get_template, text_layer, words


@dataclass(frozen=True)
class DesignSuite:
    name: str
    label: str
    collection: str
    art: str
    typography: str
    transition: str
    ink: tuple
    paper: tuple
    accent: tuple
    secondary: tuple
    dark: bool = False

    def template(self, language='zh'):
        if language not in ('zh','en'): raise ValueError('套装语言必须为 zh 或 en')
        return f'{self.typography}-{language}'

    @property
    def theme(self):
        return ArtTheme(self.art,self.label,self.ink,self.paper,(self.accent,self.secondary,self.accent))


SUITES = (
    DesignSuite('matchday','赤线竞技','MATCHDAY','manga','arena','velocity',
                (24,25,26),(246,240,224),(239,76,43),(239,76,43),True),
    DesignSuite('atelier','纸上球场','COURT ATELIER','papercut','editorial','paper',
                (48,62,51),(245,239,222),(204,117,87),(142,158,127)),
    DesignSuite('sumi','墨间回合','INK & MOTION','ink','cinema','ink',
                (37,44,41),(244,242,230),(167,69,51),(119,149,131)),
    DesignSuite('aurora','极光棱镜','PRISM COURT','clay','minimal','prism',
                (12,20,40),(233,241,250),(134,200,245),(169,158,239),True),
    DesignSuite('archive','胶片纪事','COURT ARCHIVE','retro','cinema','film',
                (65,43,32),(243,230,204),(196,101,51),(185,147,91)),
)
SUITE_IDS = ('custom',)+tuple(s.name for s in SUITES)
EXTRA_ART = APP/'assets/design_suites/aurora-ball.png'


def get_suite(name='custom'):
    if name=='custom': return None
    for suite in SUITES:
        if suite.name==name: return suite
    raise ValueError(f'未知设计套装：{name}')


def resolve_design(name='custom', language='zh', art='default', title='legacy', transition='fade'):
    """A named suite owns its three component selections; custom stays unchanged."""
    suite=get_suite(name)
    if language not in ('zh','en'): raise ValueError('套装语言必须为 zh 或 en')
    if suite: return suite.art,suite.template(language),suite.transition
    return art,title,transition


def palette(art='default', name='custom'):
    suite=get_suite(name)
    return suite.theme if suite else get_theme(art)


def suite_assets(name='custom'):
    suite=get_suite(name)
    return [Path(__file__)]+([EXTRA_ART] if suite and suite.name=='aurora' else [])


def design_manifest(name,language='zh'):
    suite=get_suite(name)
    return {} if suite is None else {**asdict(suite),'language':language,'title_template':suite.template(language),
        'motion_revision':1,'title_static_hold_sec':2.2,'hero_motion':'independent, bounded; never on match footage'}


def hero_path(item,name):
    from .illustrated import illustration_for
    suite=get_suite(name)
    if not suite: raise ValueError('需选择设计套装')
    return EXTRA_ART if name=='aurora' else illustration_for(item,suite.art)


def _fit_art(path,box):
    from PIL import Image
    with Image.open(path) as source:
        image=source.convert('RGBA')
    if image.getchannel('A').getextrema()[0]!=0: raise ValueError('装饰插图必须有真实透明通道')
    bounds=image.getchannel('A').getbbox()
    if bounds is None: raise ValueError('空白装饰插图')
    # Runtime layout only: fit the painted bounds; preserve every source PNG.
    image=image.crop(bounds)
    image.thumbnail(box,Image.Resampling.LANCZOS)
    return image


class SuiteCard:
    """Layered 720×1280 title card. Text is fixed; only the hero moves in hold.

    Regions above 480 and below 1060 are invariant during the reading interval.
    Verification reconstructs the expected animated frame, not a static proxy.
    """
    def __init__(self,item,title,top_k,name,language='zh'):
        from PIL import Image,ImageDraw,ImageOps
        from .illustrated import prepare_brand,LOGO_PNG,sticker,rank_label
        from .transitions import ASSET_ROOT
        import numpy as np
        self.suite=s=get_suite(name)
        if s is None: raise ValueError('套装标题卡不能使用 custom')
        self.template=s.template(language)
        self.name=name
        self.background=Image.new('RGBA',(720,1280),s.ink if s.dark else s.paper)
        draw=ImageDraw.Draw(self.background)
        fg=s.paper if s.dark else s.ink
        # Reuse the transition's material very quietly in the card itself.
        if name in ('atelier','sumi','archive'):
            with Image.open(ASSET_ROOT/f'{s.transition}.png') as raw:
                texture=ImageOps.fit(raw.convert('RGB'),(720,1280),method=Image.Resampling.LANCZOS).convert('RGBA')
            texture.putalpha(12 if name=='atelier' else 7)
            self.background.alpha_composite(texture)
        elif name=='aurora':
            y,x=np.mgrid[0:1280,0:720]
            glow=np.exp(-(((x-380)/300)**2+((y-800)/360)**2))*30
            pixels=np.asarray(self.background).copy()
            pixels[:,:,:3]=np.clip(pixels[:,:,:3].astype(float)+glow[:,:,None]*np.array([.55,.75,1.4]),0,255).astype('uint8')
            self.background=Image.fromarray(pixels)
        draw=ImageDraw.Draw(self.background)
        if name=='matchday':
            draw.polygon(((-100,827),(720,507),(720,686),(-100,1006)),fill=s.accent)
            draw.line((0,1103,720,820),fill=s.accent,width=2)
            draw.rectangle((48,142,102,148),fill=s.accent)
        elif name=='atelier':
            draw.rectangle((48,490,77,1044),fill=s.accent)
            draw.rectangle((500,501,660,1017),fill=s.secondary)
            draw.line((48,142,672,142),fill=s.ink,width=1)
        elif name=='sumi':
            draw.line((350,153,370,153),fill=s.accent,width=2)
            draw.ellipse((178,572,543,937),outline=s.secondary,width=1)
            draw.rectangle((616,979,656,1034),fill=s.accent)
        elif name=='aurora':
            for x in (48,672): draw.line((x,514,x,1024),fill=(40,66,93),width=1)
            for y in (520,1020):
                draw.line((48,y,81,y),fill=s.accent,width=2)
                draw.line((639,y,672,y),fill=s.accent,width=2)
        else:
            draw.rectangle((74,482,646,1050),fill=s.ink)
            draw.rectangle((89,497,631,1035),fill=(224,198,155))
            draw.line((48,145,672,145),fill=s.accent,width=2)
        # Background numbering supplies hierarchy, never competes with the label.
        number_color=tuple(round(a*.25+b*.75) for a,b in zip(s.accent,s.ink if s.dark else s.paper))
        number=text_layer(f'{item["rank"]:02d}',186,number_color,330,self.template,font_path=FONTS/'NotoSans-Bold.ttf')
        if name in ('matchday','atelier'):
            self.background.alpha_composite(number,(48 if name=='matchday' else 82,823))
        self.foreground=Image.new('RGBA',(720,1280))
        front=ImageDraw.Draw(self.foreground)
        prepare_brand()
        logo=sticker(LOGO_PNG,(216,59))
        if s.dark: front.rounded_rectangle((43,44,273,112),radius=3,fill=s.paper)
        self.foreground.alpha_composite(logo,(50,48))
        collection=text_layer(s.collection,16,fg,355,self.template,tracking=1,font_path=FONTS/'NotoSans-Bold.ttf')
        self.foreground.alpha_composite(collection,(672-collection.width,64))
        label=text_layer(rank_label(item['rank'],top_k,self.template,item.get('collection','highlights')),28,fg,610,self.template)
        centered=name in ('sumi','aurora','archive')
        self.foreground.alpha_composite(label,((720-label.width)//2 if centered else 42,172))
        lines=title.splitlines()
        if not 1<=len(lines)<=2: raise ValueError('套装标题须为一到两行')
        sizes=(100,59) if language=='zh' else (76,40)
        if name in ('sumi','archive'): sizes=(84,55) if language=='zh' else (69,39)
        for i,line in enumerate(lines):
            layer=text_layer(line,sizes[i],fg if i==0 else (s.secondary if s.dark else s.ink),624,self.template,
                             tracking=4 if name=='sumi' and language=='zh' else 0)
            if name=='matchday' and language=='zh':
                extra=round(layer.height*.10)
                layer=layer.transform((layer.width+extra,layer.height),Image.Transform.AFFINE,
                    (1,.10,-extra,0,1,0),Image.Resampling.BICUBIC)
            x=(720-layer.width)//2 if centered else 40
            self.foreground.alpha_composite(layer,(x,244 if i==0 else 376))
        front.line((48,1101,672,1101),fill=s.accent,width=1)
        caption=text_layer(words(self.template,'caption'),20,fg,320,self.template)
        self.foreground.alpha_composite(caption,(40,1122))
        count=text_layer(f'{item["rank"]:02d} / {top_k:02d}',20,fg,200,self.template,font_path=FONTS/'NotoSans-Bold.ttf')
        self.foreground.alpha_composite(count,(672-count.width,1122))
        footer=text_layer(words(self.template,'footer'),16,fg,625,self.template)
        self.foreground.alpha_composite(footer,(40,1198))
        boxes={'matchday':(570,530),'atelier':(520,508),'sumi':(468,482),'aurora':(522,522),'archive':(478,478)}
        self.hero=_fit_art(hero_path(item,name),boxes[name])
        self.center=(381 if name=='atelier' else 360,773)

    def image(self,index=45):
        from PIL import Image
        if not 0<=index<90: raise ValueError('标题动画帧索引越界')
        p=min(1.,index/11)
        settle=(1-p)**3
        t=(index-45)/30
        dx=dy=0.;scale=1.
        if self.name=='matchday': dx=-30*settle;scale=1+.015*math.sin(t*.5)
        elif self.name=='atelier': dx=18*settle;dy=15*settle+2*math.sin(t)
        elif self.name=='sumi': dy=10*settle+2*math.sin(t*.6)
        elif self.name=='aurora': dy=22*settle+6*math.sin(t*1.3);scale=1+.007*math.sin(t)
        else: scale=1+.012*math.sin(t*.5)
        size=tuple(max(1,round(v*scale)) for v in self.hero.size)
        art=self.hero if size==self.hero.size else self.hero.resize(size,Image.Resampling.LANCZOS)
        canvas=self.background.copy()
        canvas.alpha_composite(art,(round(self.center[0]+dx-art.width/2),round(self.center[1]+dy-art.height/2)))
        canvas.alpha_composite(self.foreground)
        return canvas.convert('RGB')

    def frame(self,index=45):
        import cv2
        import numpy as np
        return cv2.cvtColor(np.asarray(self.image(index)),cv2.COLOR_RGB2BGR)


def suite_headers(item,top_k,title,name,language='zh'):
    from PIL import Image,ImageDraw
    import cv2
    import numpy as np
    from .illustrated import rank_label,chapter_label
    s=get_suite(name);template=s.template(language)
    canvas=Image.new('RGBA',(720,110))
    draw=ImageDraw.Draw(canvas)
    # Compact, flat label rather than a mismatched illustrated ribbon.
    draw.rounded_rectangle((18,24,284,86),radius=2 if name!='aurora' else 9,fill=s.ink)
    draw.rectangle((18,24,23,86),fill=s.accent)
    label=text_layer(rank_label(item['rank'],top_k,template,item.get('collection','highlights')),26,s.paper,247,template)
    canvas.alpha_composite(label,(27+(247-label.width)//2,55-label.height//2))
    heading=text_layer(title.splitlines()[0],32,s.paper,396,template,outline=s.ink)
    canvas.alpha_composite(heading,(301,14))
    sub=text_layer(chapter_label(item['rank'],top_k,template),17,s.paper,340,template,outline=s.ink)
    canvas.alpha_composite(sub,(303,65))
    frames=[]
    for i in range(16):
        frame=canvas.copy();d=ImageDraw.Draw(frame)
        p=min(1.,i/12)
        eased=1-(1-p)**(4 if name=='matchday' else 3)
        d.line((309,103,309+round(355*eased),103),fill=s.accent,width=2)
        frames.append(cv2.cvtColor(np.asarray(frame),cv2.COLOR_RGBA2BGRA))
    return frames


def suite_overlay(path,kind,index,name,language='zh'):
    from PIL import Image,ImageDraw
    s=get_suite(name);template=s.template(language)
    canvas=Image.new('RGBA',(720,1280));draw=ImageDraw.Draw(canvas)
    if kind=='teaser':
        draw.rectangle((28,23,33,89),fill=s.accent)
        title=text_layer(words(template,'teaser'),39,s.paper,643,template,outline=s.ink)
        canvas.alpha_composite(title,(43,9))
        sub=text_layer(words(template,'teaser_sub'),17,s.paper,640,template,outline=s.ink)
        canvas.alpha_composite(sub,(46,67))
        text=text_layer(words(template,'teaser_bottom'),24,s.paper,555,template)
        x=(720-text.width)//2
        draw.rounded_rectangle((x-17,1181,x+text.width+17,1239),radius=3,fill=s.ink)
        draw.rectangle((x-17,1181,x-12,1239),fill=s.accent)
        canvas.alpha_composite(text,(x,1188))
    else:
        title=text_layer(words(template,'replay'),30,s.paper,200,template,outline=s.ink)
        canvas.alpha_composite(title,(29,125))
        speed=text_layer('0.67×',22,s.paper,125,template,outline=s.ink)
        canvas.alpha_composite(speed,(238,137))
        for x,sign in ((1,1),(718,-1)):
            for y,direction in ((112,1),(873,-1)):
                draw.line((x,y,x+sign*30,y),fill=s.accent,width=2)
                draw.line((x,y,x,y+direction*25),fill=s.accent,width=2)
    canvas.save(path)


def transition_sound(name):
    """Quiet, paired editorial gestures; never replace or remix match audio."""
    get_suite(name)
    noise,level,low,high={
        'matchday':('pink',.075,900,5100),
        'atelier':('pink',.040,600,2700),
        'sumi':('brown',.040,250,1600),
        'aurora':('pink',.045,1400,5800),
        'archive':('brown',.045,350,2300),
    }[name]
    return (f'anoisesrc=d=0.4:c={noise}:r=48000:a={level}:seed=42,'
            f'highpass=f={low},lowpass=f={high},afade=t=in:d=0.08,afade=t=out:st=0.14:d=0.26,'
            'asplit=2[a][b];[b]adelay=2600[b2];[a][b2]amix=inputs=2:normalize=0,apad,atrim=duration=3')
