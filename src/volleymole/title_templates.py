"""Ten bilingual, code-native title systems; source artwork remains untouched."""
from dataclasses import dataclass
from functools import lru_cache
from .common import APP

FONTS=APP/'assets/fonts'
STYLES=('editorial','arena','cinema','pop','minimal')
TEMPLATE_IDS=('legacy',)+tuple(f'{style}-{lang}' for style in STYLES for lang in ('zh','en'))
LABELS={'editorial':'刊物编辑','arena':'竞技速报','cinema':'电影片名','pop':'潮流贴纸','minimal':'极简栏目'}


@dataclass(frozen=True)
class TitleTemplate:
    style: str
    language: str

    @property
    def font(self):
        if self.language=='en':
            return FONTS/({'cinema':'NotoSerif-Regular.ttf','arena':'NotoSans-BoldItalic.ttf'}.get(self.style,'NotoSans-Bold.ttf'))
        return FONTS/({'cinema':'NotoSerifCJKsc-Regular.otf','pop':'ZCOOLKuaiLe-Regular.ttf'}.get(self.style,'NotoSansCJKsc-Bold.otf'))


def get_template(name='legacy'):
    if name not in TEMPLATE_IDS:raise ValueError(f'未知标题模板：{name}')
    if name=='legacy':return TitleTemplate('legacy','zh')
    style,language=name.rsplit('-',1)
    return TitleTemplate(style,language)


def template_assets(name='legacy'):
    template=get_template(name)
    if name=='legacy':return [APP/'title_templates.py']
    return list(dict.fromkeys([APP/'title_templates.py',template.font,FONTS/'NotoSans-Bold.ttf',
                              FONTS/'NOTICE-Noto-CJK.txt',FONTS/'NOTICE-Noto-Core.txt']))


def words(name,key):
    en=get_template(name).language=='en'
    return {
        'teaser':('精彩抢先看','THE HIGHLIGHT REEL'),
        'teaser_sub':('五次快切 · 好球马上来','FIVE QUICK CUTS / LET\'S PLAY'),
        'teaser_bottom':('好球，即将登场','HERE COMES THE ACTION'),
        'replay':('精彩回放','REPLAY'),
        'first':('精彩开场','OPENING PLAY'),
        'last':('压轴登场','FINAL PICK'),
        'next':('好球接着来','UP NEXT'),
        'footer':('日常排球，也有高光时刻','EVERYDAY VOLLEYBALL. REAL HIGHLIGHTS.'),
        'caption':('回合精选','RALLY SELECTION'),
    }[key][int(en)]


def display_title(item,name):
    """Evidence-limited editorial copy, without invented scores or winners."""
    en=get_template(name).language=='en'
    title=item['title']
    if '极低' in title:pair=('贴地救球\n看这一瞬','LOW DIG\nIN FOCUS')
    elif '低姿' in title:pair=('压低重心\n迎接来球','GET LOW\nMEET THE BALL')
    elif '低位' in title:pair=('低位防守\n这一拍','LOW DEFENSE\nIN FOCUS')
    elif '二传' in title or '组织' in title:pair=('二传组织\n衔接之间','THE SET\nCONNECTING PLAYS')
    elif '长回合' in title:pair=('长回合\n攻防之间','LONG RALLY\nBACK AND FORTH')
    elif '攻防' in title or '往返' in title:pair=('来回攻防\n继续较量','BACK AND FORTH\nTHE RALLY CONTINUES')
    else:pair=('球场时刻\n一起看球','ON COURT\nIN THE MOMENT')
    result=pair[int(en)]
    if en and get_template(name).style=='cinema':result='\n'.join(line.capitalize() for line in result.splitlines())
    return result


@lru_cache(maxsize=256)
def _font(path,size):
    from PIL import ImageFont
    return ImageFont.truetype(str(path),size)


def text_layer(text,size,fill,max_width,name,outline=None,tracking=0,font_path=None):
    """Measure actual glyph bounds, including stroke and spacing, before fitting."""
    from PIL import Image,ImageDraw
    template=get_template(name)
    path=font_path or template.font
    text=' '.join(str(text).split())
    if not text:raise ValueError('标题文字不能为空')
    if max_width<24:raise ValueError('文字可用宽度过小')
    stroke=2 if outline is not None else 0
    for fitted in range(size,3,-1):
        font=_font(str(path),fitted)
        if tracking:
            length=sum(font.getlength(c) for c in text)+tracking*(len(text)-1)
            bbox=(0,font.getbbox(text)[1],int(length)+2,font.getbbox(text)[3])
        else:bbox=font.getbbox(text,stroke_width=stroke)
        width=bbox[2]-bbox[0]+16
        if width<=max_width:break
    else:raise ValueError('标题过长，无法在安全区内排版')
    image=Image.new('RGBA',(width,bbox[3]-bbox[1]+16))
    draw=ImageDraw.Draw(image)
    x,y=8-bbox[0],8-bbox[1]
    if tracking:
        for char in text:
            draw.text((x,y),char,font=font,fill=fill,stroke_width=stroke,stroke_fill=outline)
            x+=font.getlength(char)+tracking
    else:draw.text((x,y),text,font=font,fill=fill,stroke_width=stroke,stroke_fill=outline)
    return image


def headline(title,name,ink,accent,paper,box=(608,260)):
    """A transparent typography block, with language-specific line treatments."""
    from PIL import Image,ImageDraw
    template=get_template(name)
    style,lang=template.style,template.language
    width,height=box
    if not title.strip():raise ValueError('主标题不能为空')
    canvas=Image.new('RGBA',box)
    draw=ImageDraw.Draw(canvas)
    lines=title.splitlines()
    if len(lines)==1:
        text=lines[0]
        if lang=='zh' and len(text)>8:lines=[text[:len(text)//2],text[len(text)//2:]]
        elif lang=='en' and len(text)>19 and ' ' in text:
            tokens=text.split();middle=max(1,len(tokens)//2)
            lines=[' '.join(tokens[:middle]),' '.join(tokens[middle:])]
    if len(lines)>2:raise ValueError('主标题最多两行，请精简文案')
    if style=='editorial':
        draw.rectangle((8,6,51,12),fill=accent)
        sizes=(88,64) if lang=='zh' else (74,49)
        positions=(30,143)
    elif style=='arena':sizes=(86,72) if lang=='zh' else (74,54);positions=(16,130)
    elif style=='cinema':sizes=(65,56) if lang=='zh' else (66,49);positions=(18,140)
    elif style=='pop':sizes=(86,72) if lang=='zh' else (78,52);positions=(10,135)
    else:sizes=(72,56) if lang=='zh' else (67,45);positions=(36,151)
    for index,line in enumerate(lines):
        fill=ink
        if style=='arena':fill=paper
        tracking=(5 if lang=='zh' else 1.5) if style=='cinema' else 0
        pad=36 if style in ('arena','pop') else 16
        layer=text_layer(line,sizes[index],fill,width-pad,name,tracking=tracking)
        x=8 if style in ('editorial','arena') else (width-layer.width)//2
        y=positions[index]
        if style=='arena':
            if lang=='zh':
                # An affine transform of live font glyphs, not of generated artwork.
                shear=.13;extra=round(layer.height*shear)
                layer=layer.transform((layer.width+extra,layer.height),Image.Transform.AFFINE,
                    (1,shear,-extra,0,1,0),Image.Resampling.BICUBIC)
            shadow=Image.new('RGBA',layer.size,accent);shadow.putalpha(layer.getchannel('A'))
            canvas.alpha_composite(shadow,(x+4,y+4))
        elif style=='pop':
            from PIL import ImageFilter
            expanded=layer.getchannel('A').filter(ImageFilter.MaxFilter(11))
            backing=Image.new('RGBA',layer.size,paper);backing.putalpha(expanded)
            canvas.alpha_composite(backing,(x,y))
            shadow=Image.new('RGBA',layer.size,accent);shadow.putalpha(expanded)
            canvas.alpha_composite(shadow,(x+4,y+5))
            canvas.alpha_composite(backing,(x,y))
        canvas.alpha_composite(layer,(x,y))
    if style=='cinema':draw.line((width//2-30,119,width//2+30,119),fill=accent,width=1)
    if style=='minimal':
        draw.line((width//2-18,14,width//2+18,14),fill=accent,width=3)
        draw.line((width//2-55,132,width//2+55,132),fill=ink,width=1)
    return canvas


def title_card(item,title,font_path,top_k,art_theme,name):
    from PIL import Image,ImageDraw
    from .art_themes import get_theme
    from .illustrated import illustration_for,sticker,prepare_brand,LOGO_PNG,rank_label
    template=get_template(name);style=template.style
    theme=get_theme(art_theme);ink,paper,accent=theme.ink,theme.cream,theme.colors[0]
    dark=style=='arena'
    canvas=Image.new('RGBA',(720,1280),(ink if dark else paper)+(255,))
    draw=ImageDraw.Draw(canvas)
    foreground=paper if dark else ink
    if style=='editorial':
        draw.line((56,145,664,145),fill=ink,width=2)
        draw.rectangle((566,223,664,669),fill=accent)
    elif style=='arena':
        draw.polygon(((0,228),(720,132),(720,228),(0,324)),fill=accent)
        draw.line((42,1086,666,1003),fill=accent,width=5)
    elif style=='cinema':
        draw.rectangle((26,25,693,1254),outline=accent,width=1)
        draw.line((286,225,434,225),fill=accent,width=1)
    elif style=='pop':
        draw.ellipse((65,268,675,830),fill=accent)
        draw.ellipse((589,227,613,251),fill=ink)
        draw.line((77,260,102,292),fill=ink,width=5)
    else:
        draw.line((56,145,664,145),fill=ink,width=1)
        draw.rectangle((332,247,388,251),fill=accent)
    prepare_brand()
    logo=sticker(LOGO_PNG,(300,82))
    if dark:draw.rounded_rectangle((42,35,374,135),radius=8,fill=paper)
    logo_x=(720-logo.width)//2 if style in ('cinema','pop','minimal') else 54
    canvas.alpha_composite(logo,(logo_x,43))
    label=rank_label(item['rank'],top_k,name,item.get('collection','highlights'))
    label_layer=text_layer(label,28,foreground,595,name)
    x=(720-label_layer.width)//2 if style in ('cinema','pop','minimal') else 52
    canvas.alpha_composite(label_layer,(x,162 if not dark else 150))
    art=sticker(illustration_for(item,art_theme),(572,476))
    canvas.alpha_composite(art,((720-art.width)//2,288+(476-art.height)//2))
    block=headline(title,name,ink,accent,paper)
    canvas.alpha_composite(block,(56,799))
    caption=text_layer(words(name,'caption'),18,foreground,580,name,
                       font_path=FONTS/'NotoSans-Bold.ttf' if template.language=='en' else None)
    canvas.alpha_composite(caption,((720-caption.width)//2,1110))
    footer=text_layer(words(name,'footer'),16,foreground,608,name)
    canvas.alpha_composite(footer,((720-footer.width)//2,1201))
    return canvas.convert('RGB')
