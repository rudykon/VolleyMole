"""Generated sports artwork and playful lettering for readable title interludes."""
from pathlib import Path
import subprocess
from .common import APP, ROOT, digest, read_json, save_json

ART_ROOT=APP/'assets/illustrated'
TITLE_FONT=APP/'assets/fonts/ZCOOLKuaiLe-Regular.ttf'
LOGO_SOURCE=APP/'assets/branding/volleymole.svg'
LOGO_PNG=APP/'assets/branding/volleymole.png'
ART_NAMES=('volley_ball','volley_save','volley_set','volley_dive','volley_receive','volley_spike','rank_ribbon')
INK=(13,22,40)
CREAM=(255,246,223)
COLORS=[(255,198,70),(85,231,199),(255,115,142)]
TRANSITION_FRAMES=90
TRANSITION_RAMP_FRAMES=12


def asset_paths():
    return [TITLE_FONT,LOGO_SOURCE,LOGO_PNG,APP/'rasterize_logo.py']+[ART_ROOT/f'{name}.png' for name in ART_NAMES]


def prepare_brand():
    """Use the exact SVG; invalidate its derived PNG when the source changes."""
    metadata=LOGO_PNG.with_suffix('.json')
    signature={'source_sha256':digest(LOGO_SOURCE),'renderer_sha256':digest(APP/'rasterize_logo.py')}
    previous=read_json(metadata) if metadata.is_file() else {}
    if (LOGO_PNG.is_file() and all(previous.get(k)==v for k,v in signature.items())
            and previous.get('png_sha256')==digest(LOGO_PNG)):
        return
    LOGO_PNG.parent.mkdir(parents=True,exist_ok=True)
    subprocess.run(['/usr/bin/python3',str(APP/'rasterize_logo.py'),str(LOGO_SOURCE),str(LOGO_PNG)],check=True)
    save_json(metadata,{**signature,'png_sha256':digest(LOGO_PNG)})


def illustration_for(item):
    title=item['title']
    if '长回合' in title or '起跳' in title:return ART_ROOT/'volley_spike.png'
    if '极低' in title:return ART_ROOT/'volley_dive.png'
    if '低位' in title:return ART_ROOT/'volley_receive.png'
    if any(word in title for word in ('低姿','低位','极低','救球','接球')):return ART_ROOT/'volley_save.png'
    if any(word in title for word in ('二传','组织','配合')):return ART_ROOT/'volley_set.png'
    return ART_ROOT/'volley_ball.png'


def rank_label(rank,top_k):
    if top_k not in (5,10) or not 1<=rank<=top_k:
        raise ValueError('名次必须在五佳球或十佳球范围内')
    chinese=('','一','二','三','四','五','六','七','八','九','十')
    return f'{chinese[top_k]}佳球 · 第{chinese[rank]}球'


def rank_badge(rank,top_k,width):
    badge=sticker(ART_ROOT/'rank_ribbon.png',(width,round(width/3)))
    label=lettering(rank_label(rank,top_k),round(width*.086),INK,round(badge.width*.71))
    badge.alpha_composite(label,((badge.width-label.width)//2+round(badge.width*.025),
                                 (badge.height-label.height)//2))
    return badge


def chapter_label(rank,top_k):
    if rank==1:return '压轴登场'
    if rank==top_k:return '精彩开场'
    return '好球接着来'


def title_lines(title):
    for mark in ('，','！'):
        if mark in title[:-1]:
            a,b=title.split(mark,1)
            return [a+('！' if mark=='！' else ''),b]
    if len(title)>9:return [title[:len(title)//2],title[len(title)//2:]]
    return [title]


def transition_opacity(frame,total=TRANSITION_FRAMES):
    ramp=TRANSITION_RAMP_FRAMES
    p=min(1.,frame/(ramp-1),(total-1-frame)/(ramp-1))
    p=max(0.,p)
    return p*p*(3-2*p)


def sticker(path,box):
    from PIL import Image
    if not Path(path).is_file():raise ValueError(f'缺失生图插画：{path}')
    art=Image.open(path).convert('RGBA')
    if art.getchannel('A').getextrema()[0]==255:
        raise ValueError('插画必须保留生成素材的透明通道')
    art.thumbnail(box,Image.Resampling.LANCZOS)
    return art


def lettering(text,size,fill,max_width,tilt=0,outline=None):
    from PIL import Image,ImageDraw,ImageFont
    if not TITLE_FONT.is_file():raise ValueError('缺失站酷快乐体标题字体')
    font=ImageFont.truetype(str(TITLE_FONT),size)
    while font.getlength(text)>max_width-16:
        size-=1;font=ImageFont.truetype(str(TITLE_FONT),size)
    stroke=3 if outline is not None else 1
    bbox=font.getbbox(text,stroke_width=max(2,stroke))
    layer=Image.new('RGBA',(bbox[2]-bbox[0]+18,bbox[3]-bbox[1]+18))
    draw=ImageDraw.Draw(layer)
    draw.text((9-bbox[0]+2,9-bbox[1]+3),text,font=font,fill=(6,12,25,90))
    draw.text((9-bbox[0],9-bbox[1]),text,font=font,fill=fill,stroke_width=stroke,
              stroke_fill=outline if outline is not None else fill)
    if tilt:layer=layer.rotate(tilt,resample=Image.Resampling.BICUBIC,expand=True)
    if layer.width>max_width:
        layer.thumbnail((max_width,layer.height),Image.Resampling.LANCZOS)
    return layer


def headers(item,font_path,top_k,title):
    """Return BGRA title layers; empty pixels reveal the same live video frame."""
    import cv2
    import numpy as np
    from PIL import Image,ImageDraw,ImageFont
    color=COLORS[(item['rank']-1)%len(COLORS)]
    art=sticker(illustration_for(item),(64,61))
    badge=rank_badge(item['rank'],top_k,290)
    line=lettering(title,36,CREAM,412,tilt=1,outline=INK)
    sub=ImageFont.truetype(str(font_path),18)
    frames=[]
    for i in range(16):
        p=min(1.,i/12)
        canvas=Image.new('RGBA',(720,110))
        draw=ImageDraw.Draw(canvas)
        draw.line((305,100,483,103,614,99),fill=INK,width=6)
        draw.line((305,100,483,103,614,99),fill=color,width=3)
        # Ranking stays fully visible from the first frame; only the artwork moves.
        canvas.alpha_composite(badge,(4,(110-badge.height)//2))
        canvas.alpha_composite(art,(644+round(65*(1-p)**3),49))
        canvas.alpha_composite(line,(298,3))
        draw.text((310,73),chapter_label(item['rank'],top_k),font=sub,fill=color,
                  stroke_width=2,stroke_fill=INK)
        frames.append(cv2.cvtColor(np.asarray(canvas),cv2.COLOR_RGBA2BGRA))
    return frames


def composite_header(background,layer):
    """Composite only the code-native UI layer, without flattening its alpha."""
    import numpy as np
    if background.shape!=layer.shape[:2]+(3,) or layer.shape[2]!=4:
        raise ValueError('透明标题与视频区域尺寸不匹配')
    alpha=layer[:,:,3:4].astype(np.float32)/255
    return np.rint(layer[:,:,:3]*alpha+background*(1-alpha)).astype(np.uint8)


def overlay(path,kind,font_path,index=0):
    from PIL import Image,ImageDraw,ImageFont
    canvas=Image.new('RGBA',(720,1280))
    draw=ImageDraw.Draw(canvas)
    color=COLORS[index%len(COLORS)]
    if kind=='teaser':
        # Let the opening highlight stay visible behind its title; artwork and
        # outlined lettering are the only pixels added at the top.
        teasers=('volley_ball','volley_dive','volley_set','volley_receive','volley_spike')
        art=sticker(ART_ROOT/f'{teasers[index%len(teasers)]}.png',(115,106))
        canvas.alpha_composite(art,(3,2))
        canvas.alpha_composite(lettering('先看这几下！',48,color,570,1,outline=INK),(125,1))
        draw.text((141,76),'精彩抢先看  ·  好球马上来',font=ImageFont.truetype(str(font_path),19),fill=CREAM,
                  stroke_width=2,stroke_fill=INK)
        draw.rounded_rectangle((185,1180,535,1245),radius=24,fill=INK+(240,))
        text=lettering('别眨眼，好球来了！',30,CREAM,325)
        canvas.alpha_composite(text,((720-text.width)//2,1182))
    else:
        # Transparent replay caption: only glyphs/outline cover the live picture.
        text=lettering('再看一次',37,(255,198,70),207,-2,outline=INK)
        canvas.alpha_composite(text,(29,125))
        draw.text((228,145),'0.67×',font=ImageFont.truetype(str(font_path),28),fill=CREAM,
                  stroke_width=2,stroke_fill=INK)
        draw.rectangle((0,111,719,874),outline=(255,198,70,255),width=4)
    canvas.save(path)


def title_card(item,title,font_path,top_k):
    from PIL import Image,ImageDraw,ImageFont
    canvas=Image.new('RGBA',(720,1280),CREAM+(255,))
    draw=ImageDraw.Draw(canvas)
    color=COLORS[(item['rank']-1)%len(COLORS)]
    # A full-screen illustrated card, not an overlay competing with live action.
    draw.ellipse((-140,-165,465,385),fill=color+(255,))
    draw.polygon([(0,1174),(720,1100),(720,1280),(0,1280)],fill=INK+(255,))
    draw.arc((569,148,650,229),-35,230,fill=INK,width=5)
    draw.line((636,269,676,284),fill=INK,width=5)
    draw.line((60,644,92,629),fill=INK,width=5)
    small=ImageFont.truetype(str(font_path),21)
    prepare_brand()
    logo=sticker(LOGO_PNG,(490,134))
    canvas.alpha_composite(logo,((720-logo.width)//2,21))
    badge=rank_badge(item['rank'],top_k,620)
    canvas.alpha_composite(badge,((720-badge.width)//2,151))
    art=sticker(illustration_for(item),(580,430))
    canvas.alpha_composite(art,((720-art.width)//2,348+(430-art.height)//2))
    lines=title_lines(title)
    y=785 if len(lines)>1 else 820
    for i,line in enumerate(lines):
        text=lettering(line,68,INK,642,tilt=2 if i==0 else -1)
        canvas.alpha_composite(text,((720-text.width)//2,y+i*102))
    draw.line((188,999,332,1004,532,996),fill=color,width=10)
    caption=f'{chapter_label(item["rank"],top_k)}  /  接着看这一球'
    draw.text(((720-draw.textlength(caption,font=small))/2,1040),caption,font=small,fill=INK)
    draw.text((205,1173),'日常排球，也有高光时刻',font=small,fill=CREAM)
    return canvas.convert('RGB')


def encode_transition(previous,following,path,segment,font_path):
    import cv2
    import numpy as np
    from .media_worker import frame_at
    before=frame_at(previous['path'],max(0,previous['duration_sec']-1/30),720)
    after=frame_at(following['path'],0,720)
    item={'rank':segment['next_rank'],'title':segment['original_title']}
    card=title_card(item,segment['display_title'],font_path,segment['top_k'])
    # Save the actual, human-readable card as an auditable design asset.
    card.save(path.with_suffix('.png'))
    card=cv2.cvtColor(np.asarray(card),cv2.COLOR_RGB2BGR)
    duration=segment['duration_sec']
    sound=f'anoisesrc=d={duration}:c=pink:r=48000:a=0.08:seed=42,highpass=f=800,lowpass=f=4500,afade=t=in:d=0.10,afade=t=out:st=0.15:d=0.30'
    process=subprocess.Popen(['ffmpeg','-y','-v','error','-f','rawvideo','-pix_fmt','bgr24','-s','720x1280','-r','30','-i','pipe:0',
        '-f','lavfi','-i',sound,'-t',str(duration),'-c:v','libx264','-preset','fast','-crf','20','-threads','4',
        '-pix_fmt','yuv420p','-c:a','aac','-ar','48000','-ac','2','-b:a','192k',str(path)],stdin=subprocess.PIPE)
    try:
        for i in range(segment['output_frames']):
            alpha=transition_opacity(i,segment['output_frames'])
            base=before if i<segment['output_frames']//2 else after
            frame=cv2.addWeighted(card,alpha,base,1-alpha,0)
            process.stdin.write(frame.tobytes())
        process.stdin.close()
        if process.wait():raise RuntimeError('插画标题转场编码失败')
    finally:
        if process.poll() is None:process.terminate();process.wait()
