"""Traceable, unaltered source views for temporal replay decisions.

The complete frame always remains visible. A detail panel is supplementary and
uses the very same decoded PTS; it is never a tracked substitute for a panorama.
"""
import base64
import copy
import math
import time


def _decode(part):
    import cv2
    import numpy as np
    url = part['image_url']['url']
    if not url.startswith('data:image/'):
        raise ValueError('replay_frame_label_url')
    pixels = cv2.imdecode(np.frombuffer(base64.b64decode(url.split(',', 1)[1]), dtype=np.uint8), cv2.IMREAD_COLOR)
    if pixels is None:
        raise ValueError('replay_frame_label_decode')
    return pixels


def _encode(pixels):
    import cv2
    ok, encoded = cv2.imencode('.jpg', pixels, [cv2.IMWRITE_JPEG_QUALITY, 90])
    if not ok:
        raise ValueError('replay_frame_label_encode')
    return 'data:image/jpeg;base64,' + base64.b64encode(encoded).decode()


def _caption(pixels, text):
    import cv2
    border = max(28, round(pixels.shape[1] * .045))
    labelled = cv2.copyMakeBorder(pixels, border, 0, 0, 0, cv2.BORDER_CONSTANT, value=(18, 18, 18))
    # Fit long source timestamps/identifiers without writing over the picture.
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = max(.4, pixels.shape[1] / 1350)
    scale = min(scale, (pixels.shape[1]-16) / max(1, cv2.getTextSize(text, font, 1., 1)[0][0]))
    cv2.putText(labelled, text, (8, border-8), font, scale, (255, 255, 255), 1, cv2.LINE_AA)
    return labelled


def _frames_and_images(evidence, content):
    frames = [r for r in evidence if r['kind'] == 'frame']
    images = [r for r in content if r['type'] == 'image_url']
    if len(frames) != len(images):
        raise ValueError('replay_frame_label_count')
    return frames, images


def label_frames(evidence, content):
    """Legacy in-place API: add a border, never cover a ball or player."""
    frames, images = _frames_and_images(evidence, content)
    for frame, part in zip(frames, images):
        caption = f"{frame['id']}   source_sec={frame['start_sec']:.6f}"
        part['image_url']['url'] = _encode(_caption(_decode(part), caption))
    return content


def _indices(count, limit):
    """Evenly spaced indices include both boundaries, with no duplicate cells."""
    if limit <= 0:
        return []
    if count <= limit:
        return list(range(count))
    if limit == 1:
        return [count // 2]
    return [round(i * (count-1) / (limit-1)) for i in range(limit)]


def _motion_box(pixels):
    """Conservative spatial hint from the current sequence, never an action label.

    Global camera movement, near-static scenes, and nonlocal motion produce no
    crop. The panorama still supplies the entire ball trajectory when a tiny
    moving ball contributes too little energy to this estimate.
    """
    import cv2
    import numpy as np
    if len(pixels) < 3:
        return None
    shapes = {p.shape for p in pixels}
    if len(shapes) != 1:
        return None
    height, width = pixels[0].shape[:2]
    small_width = min(320, width)
    small_height = max(2, round(height * small_width / width))
    gray = [cv2.cvtColor(cv2.resize(p, (small_width, small_height)), cv2.COLOR_BGR2GRAY)
            for p in pixels]
    diffs = [cv2.absdiff(a, b) for a, b in zip(gray, gray[1:])]
    # Dense changes usually mean a pan, a cut, flicker, or a moving foreground.
    if np.median([np.mean(d > 18) for d in diffs]) > .28:
        return None
    energy = np.sum([np.where(d > 18, d.astype(float)-18, 0) for d in diffs], axis=0)
    if np.count_nonzero(energy) < max(12, small_width*small_height*.001):
        return None
    def bounds(weights):
        cumulative = np.cumsum(weights)
        return (int(np.searchsorted(cumulative, cumulative[-1]*.01)),
                int(np.searchsorted(cumulative, cumulative[-1]*.99))+1)
    x0, x1 = bounds(energy.sum(axis=0)); y0, y1 = bounds(energy.sum(axis=1))
    box = _padded_box((x0/small_width, y0/small_height, x1/small_width, y1/small_height))
    if ((box[2]-box[0])*(box[3]-box[1]) >= .82
            or box[2]-box[0] >= 1/1.12):
        return None
    return box


def _padded_box(box):
    x0, y0, x1, y1 = box
    # Keep surrounding player and ball context, not just a hand-sized crop.
    cx = (x0+x1)/2; cy = (y0+y1)/2
    width = min(1., max(.45, x1-x0+.16))
    height = min(1., max(.40, y1-y0+.16))
    x0 = max(0., min(1.-width, cx-width/2))
    y0 = max(0., min(1.-height, cy-height/2))
    return (x0, y0, x0+width, y0+height)


def _hint_box(hints, when):
    active = []
    for hint in hints:
        if hint.get('coordinate_space') != 'normalized_display':
            raise ValueError('replay_visual_hint_coordinates')
        box = hint.get('bbox')
        interval = (hint.get('start_sec'), hint.get('end_sec'))
        if (not isinstance(box, (tuple, list)) or len(box) != 4
                or any(type(v) not in (int, float) or not math.isfinite(v) for v in (*box, *interval))
                or not 0 <= box[0] < box[2] <= 1 or not 0 <= box[1] < box[3] <= 1
                or interval[0] > interval[1]):
            raise ValueError('replay_visual_hint_bounds')
        if interval[0]-1e-6 <= when <= interval[1]+1e-6:
            active.append(box)
    if not active:
        return None
    # Union simultaneous hints: never privilege one player over a visible ball.
    return _padded_box((min(b[0] for b in active), min(b[1] for b in active),
                        max(b[2] for b in active), max(b[3] for b in active)))


def _detail_plan(frames, hints, limit):
    """Prioritize consecutive source frames where spatial hints are available.

    Every sufficiently covered hint region receives at least three adjacent
    sampled frames when the budget permits. In the usual single-focus case a
    seven-frame burst exposes preparation/contact/departure. Remaining slots
    preserve broad temporal context. Excess or sparse regions are explicit in
    the audit; missing evidence is never invented to claim a complete burst.
    """
    import statistics
    fallback = _indices(len(frames), limit)
    audit = dict(method='uniform', budget=limit, regions=[], selected_frame_ids=[])
    if not hints or not limit:
        audit['selected_frame_ids'] = [frames[i]['id'] for i in fallback]
        return fallback, audit
    times = [frame['start_sec'] for frame in frames]
    step = statistics.median(b-a for a, b in zip(times, times[1:])) if len(times)>1 else 0.
    active = [i for i,t in enumerate(times) if _hint_box(hints,t) is not None]
    runs = []
    for index in active:
        if (not runs or index != runs[-1][-1]+1
                or times[index]-times[runs[-1][-1]] > step*1.5+1e-6):
            runs.append([])
        runs[-1].append(index)
    viable = [r for r in runs if len(r)>=3]
    # If there are more regions than the budget can bracket, prefer those with
    # more actual temporal evidence and explicitly report the unserved regions.
    supported = sorted(viable,key=lambda r:(-len(r),times[r[0]]))[:limit//3]
    supported = sorted(supported,key=lambda r:times[r[0]])
    chosen = set()
    target = min(7,limit//len(supported)) if supported else 0
    for run in runs:
        row = dict(start_sec=times[run[0]],end_sec=times[run[-1]],available_frame_count=len(run),frame_ids=[])
        if run not in supported:
            row.update(status='deferred',reason='fewer_than_three_consecutive_hint_frames' if len(run)<3
                       else 'detail_budget_cannot_bracket_every_hint_region')
        else:
            count = min(target,len(run))
            center = (times[run[0]]+times[run[-1]])/2
            anchor = min(range(len(run)),key=lambda k:abs(times[run[k]]-center))
            start = max(0,min(len(run)-count,anchor-(count-1)//2))
            burst = run[start:start+count]
            chosen.update(burst)
            row.update(status='continuous_burst',frame_ids=[frames[i]['id'] for i in burst])
        audit['regions'].append(row)
    if supported:
        audit['method'] = 'hint_prioritized_continuous_bursts'
        remaining = min(limit,len(frames))-len(chosen)
        for index in _indices(len(frames),remaining):
            chosen.add(index)
        while len(chosen)<min(limit,len(frames)):
            chosen.add(max((i for i in range(len(frames)) if i not in chosen),
                           key=lambda i:min(abs(times[i]-times[j]) for j in chosen)))
        selected = sorted(chosen)
    else:
        selected = fallback
    audit['selected_frame_ids'] = [frames[i]['id'] for i in selected]
    return selected,audit


def _native_frames(source, frames, deadline):
    """Decode requested *actual* PTS, rejecting nearest-frame substitutions."""
    import av
    import cv2
    if not frames:
        return {}
    origin = source.get('start_sec', 0.)
    times = [r['start_sec'] for r in frames]
    out = {}; index = 0
    with av.open(source['path']) as container:
        stream = container.streams.video[0]
        container.seek(max(0, int((times[0]+origin)*av.time_base)), backward=True)
        for frame in container.decode(stream):
            if deadline is not None and time.monotonic() >= deadline:
                raise TimeoutError('replay_visual_context_deadline')
            if frame.pts is None:
                continue
            when = float(frame.pts*frame.time_base)-origin
            if when < times[index]-1e-6:
                continue
            if abs(when-times[index]) > 1e-6:
                raise ValueError('replay_visual_pts_mismatch')
            pixels = frame.to_ndarray(format='bgr24')
            rotation = source.get('rotation', 0)
            if rotation == 90:
                pixels = cv2.rotate(pixels, cv2.ROTATE_90_COUNTERCLOCKWISE)
            elif rotation == 180:
                pixels = cv2.rotate(pixels, cv2.ROTATE_180)
            elif rotation == 270:
                pixels = cv2.rotate(pixels, cv2.ROTATE_90_CLOCKWISE)
            elif rotation != 0:
                raise ValueError('replay_visual_rotation')
            out[frames[index]['id']] = pixels
            index += 1
            if index == len(times):
                break
    if index != len(times):
        raise ValueError('replay_visual_pts_missing')
    return out


def prepare_visual_context(evidence, content, source=None, *, spatial_hints=None,
                           overview_frames=12, detail_limit=12, deadline=None):
    """Return ``(content, audit)`` containing panoramas, overview, native details.

    ``evidence`` is unchanged. Every image/cell shows an existing frame ID and
    source-local PTS. ``content`` is copied. Pass clean sampler output, not output
    from ``label_frames``. Optional spatial hints are dictionaries with
    ``start_sec``, ``end_sec``, ``bbox=[left,top,right,bottom]``, and
    ``coordinate_space='normalized_display'`` (after source rotation).

    Without hints, motion from this exact sequence supplies a conservative crop.
    Detail panels require ``source`` and decode its original-resolution pixels;
    resizing a thumbnail is never presented as recovered visual detail. Set
    ``overview_frames=0`` or ``detail_limit=0`` to disable the respective views.
    """
    import cv2
    import numpy as np
    content = copy.deepcopy(content)
    frames, images = _frames_and_images(evidence, content)
    times = [r['start_sec'] for r in frames]
    if (not frames or len({r['id'] for r in frames}) != len(frames)
            or any(type(t) not in (int, float) or not math.isfinite(t) for t in times)
            or any(b <= a for a, b in zip(times, times[1:]))):
        raise ValueError('replay_visual_frame_order')
    if (type(overview_frames) is not int or not 0 <= overview_frames <= 24
            or type(detail_limit) is not int or not 0 <= detail_limit <= 96):
        raise ValueError('replay_visual_budget')
    pixels = [_decode(part) for part in images]
    if spatial_hints:
        _hint_box(spatial_hints, times[0])
    motion_box = _motion_box(pixels) if source and detail_limit else None
    boxes = {}
    detail_indices, detail_selection = _detail_plan(frames, spatial_hints or [], detail_limit if source else 0)
    for index in detail_indices:
        box = _hint_box(spatial_hints or [], times[index]) or motion_box
        if box and (box[2]-box[0])*(box[3]-box[1]) < .90:
            boxes[index] = box
    native = _native_frames(source, [frames[i] for i in boxes], deadline) if boxes else {}
    audit = dict(version=2, time_basis='source-local actual PTS', frame_count=len(frames),
                 overview=[], details=[], motion_bbox=motion_box, detail_selection=detail_selection,
                 coordinate_space='normalized_display_after_source_rotation',
                 source_sha256=(source or {}).get('identity', {}).get('sha256'),
                 motion_roi_checks=dict(usable_for_supplementary_crop=motion_box is not None,
                    basis='bounded_local_temporal_difference',
                    is_action_or_ball_detection=False,
                    panoramic_context_required=True))
    for index, (frame, part, panorama) in enumerate(zip(frames, images, pixels)):
        caption = f"{frame['id']} source_sec={frame['start_sec']:.6f} FULL"
        composed = _caption(panorama, caption)
        if index in boxes:
            original = native[frame['id']]
            height, width = original.shape[:2]; x0, y0, x1, y1 = boxes[index]
            left, top = math.floor(x0*width), math.floor(y0*height)
            right, bottom = math.ceil(x1*width), math.ceil(y1*height)
            crop = original[top:bottom, left:right]
            # A low-resolution original cannot gain information through zoom.
            scale = panorama.shape[1] / crop.shape[1]
            full_scale = panorama.shape[1] / width
            if width > panorama.shape[1] and scale > full_scale * 1.12 and crop.shape[1] >= panorama.shape[1]//2:
                detail = cv2.resize(crop, (panorama.shape[1], max(2, round(crop.shape[0]*scale))), interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)
                detail = _caption(detail, f"{frame['id']} {frame['start_sec']:.6f}s SAME FRAME DETAIL")
                composed = np.vstack((composed, detail))
                audit['details'].append(dict(frame_id=frame['id'], source_sec=frame['start_sec'],
                    bbox_normalized=list(boxes[index]), crop_pixels=[left, top, right, bottom],
                    original_size=[width, height], method='spatial_hint' if _hint_box(spatial_hints or [], frame['start_sec']) else 'sequence_motion'))
        part['image_url']['url'] = _encode(composed)
        part['image_url']['detail'] = 'high'
    selected = _indices(len(frames), overview_frames)
    if selected:
        cells = []
        cell_width = 384
        # Common cell size uses letterboxing, without cropping any source frame.
        cell_height = max(round(pixels[i].shape[0]*cell_width/pixels[i].shape[1]) for i in selected)
        for index in selected:
            frame = frames[index]; source_pixels = pixels[index]
            height = max(2, round(source_pixels.shape[0]*cell_width/source_pixels.shape[1]))
            scaled = cv2.resize(source_pixels, (cell_width, height), interpolation=cv2.INTER_AREA)
            cell = np.full((cell_height, cell_width, 3), 18, dtype=np.uint8)
            cell[:height] = scaled
            cells.append(_caption(cell, f"{frame['id']} {frame['start_sec']:.6f}s"))
            audit['overview'].append(dict(frame_id=frame['id'], source_sec=frame['start_sec']))
        columns = min(3, len(cells))
        blank = np.full_like(cells[0], 18)
        while len(cells) % columns:
            cells.append(blank)
        sheet = np.vstack([np.hstack(cells[i:i+columns]) for i in range(0, len(cells), columns)])
        content[:0] = [dict(type='text', text='CHRONOLOGICAL OVERVIEW: left to right, then next row. These are sparse navigation frames, not extra evidence. Use the following complete sequence to identify contact or completion. DETAIL panels below panoramas show the SAME frame/PTS; always inspect the full view for the complete ball path.'),
                       dict(type='image_url', image_url=dict(url=_encode(sheet), detail='high'))]
    return content, audit
