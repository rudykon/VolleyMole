"""Conservative extra rally evidence; never turn an interpolated point into a detection."""
import numpy as np


def preserve_separations(groups, airborne_cores, edges, min_core_sec):
    """New low/auxiliary evidence cannot erase a separation between strong cores.

    The midpoint is a conservative boundary hypothesis, not a detected point or
    a confirmed end of play. Short occlusions inside each core remain bridged.
    """
    strong = [g for g in airborne_cores if edges[g[-1]+1]-edges[g[0]]>=min_core_sec]
    cuts = sorted({(left[-1]+1+right[0])//2 for left,right in zip(strong,strong[1:])})
    result,applied = [],[]
    for group in groups:
        parts = [group]
        for cut in cuts:
            if not group[0]<cut<=group[-1]:
                continue
            applied.append(float(edges[cut]))
            parts = [part for old in parts for part in
                     ([i for i in old if i<cut],[i for i in old if i>=cut]) if part]
        result.extend(parts)
    return result,sorted(set(applied))


def associate_auxiliary(records, xy, visible, times, width, confidence=.65, max_gap=.65):
    """Accept actual auxiliary detections only between plausible VballNet anchors.

    Anchors remain raw VballNet observations. At least two supported auxiliary
    observations in a local gap are required; single false positives are retained
    in the analytics JSON but not promoted into the camera track.
    """
    raw_visible = visible.copy()
    valid = np.flatnonzero(raw_visible)
    proposal = {}
    if len(valid)<2:
        return xy.copy(), visible.copy(), ['vball' if v else 'missing' for v in visible], []
    for i,row in enumerate(records):
        if visible[i] or not row.get('ball'):
            continue
        ball = row['ball']
        if ball.get('confidence',0)<confidence:
            continue
        box = np.asarray(ball.get('xyxy',[]),dtype=float)
        if box.shape!=(4,) or not np.isfinite(box).all() or np.any(box[2:]<=box[:2]):
            continue
        j = np.searchsorted(valid,i)
        if not 0<j<len(valid):
            continue
        left,right = valid[j-1],valid[j]
        duration = times[right]-times[left]
        if duration>max_gap or duration<=0:
            continue
        point = (box[:2]+box[2:])/2
        a,b = times[i]-times[left], times[right]-times[i]
        if a<=0 or b<=0:
            continue
        if max(np.linalg.norm(point-xy[left])/a,np.linalg.norm(xy[right]-point)/b)>2.1*width:
            continue
        predicted = xy[left]+(xy[right]-xy[left])*a/duration
        if np.linalg.norm(point-predicted)>.06*width:
            continue
        proposal[i] = (point,int(left),int(right),ball.get('origin','auxiliary_ball_detector'))
    groups = {}
    for i,(_,left,right,_) in proposal.items():
        groups.setdefault((left,right),[]).append(i)
    fused = xy.copy()
    seen = visible.copy()
    origin = ['vball' if v else 'missing' for v in visible]
    accepted = []
    for (left,right),indices in groups.items():
        if len(indices)<2:
            continue
        for i in indices:
            fused[i],seen[i],origin[i] = proposal[i][0],True,proposal[i][3]
            accepted.append({'frame':i,'time_sec':float(times[i]),'source':origin[i],
                             'vball_anchors':[left,right],'uncertainty':'cross-model association, not calibrated identity'})
    return fused,seen,origin,accepted


def low_motion(records, xy, continuous, speed, heads, times, width, height, min_speed=.12):
    """Fast, court-local low ball supported by state/action, excluding carried motion."""
    low = np.zeros(len(times),dtype=bool)
    carried = np.zeros(len(times),dtype=bool)
    nearest = np.full_like(xy,np.nan)
    state_support = np.array([r['state'] in ('play','service') and r.get('state_confidence',.5)>=.6 for r in records])
    action_support = np.zeros(len(times),dtype=bool)
    court_local = np.zeros(len(times),dtype=bool)
    near_body = np.zeros(len(times),dtype=bool)
    for i,row in enumerate(records):
        if not continuous[i]:
            continue
        people = np.array([p['xyxy'] for p in row.get('players',[]) if p.get('confidence',0)>=.4],dtype=float)
        if len(people)<2:
            continue
        centers = np.column_stack(((people[:,0]+people[:,2])/2,people[:,1]+.45*(people[:,3]-people[:,1])))
        j = np.argmin(np.linalg.norm(centers-xy[i],axis=1))
        nearest[i] = centers[j]
        near_body[i] = np.linalg.norm(centers[j]-xy[i])<.55*(people[j,3]-people[j,1])
        court_local[i] = (max(.02*width,people[:,0].min()-.12*width)<xy[i,0]<min(.98*width,people[:,2].max()+.12*width)
                          and heads[i]<=xy[i,1]<=min(.94*height,people[:,3].max()+.04*height))
        for action in row.get('raw_actions',row.get('actions',[])):
            if action.get('class') not in ('receive','set','spike','block','serve') or action.get('confidence',0)<.4:
                continue
            box = action.get('xyxy')
            if box is not None and box[0]-.12*width<=xy[i,0]<=box[2]+.12*width and box[1]-.12*height<=xy[i,1]<=box[3]+.12*height:
                action_support[i] = True
                break
    dt = np.diff(times,prepend=times[0]-.0333)
    relative = xy-nearest
    relative_speed = np.linalg.norm(np.diff(relative,axis=0,prepend=relative[:1]),axis=1)/width/dt
    carried = near_body & np.roll(near_body,1) & np.isfinite(relative_speed) & (relative_speed<.05)
    carried[0] = False
    # Brief action evidence supports nearby frames, not arbitrary long gaps.
    hits = np.flatnonzero(action_support)
    for i in hits:
        a,b = np.searchsorted(times,[times[i]-.3,times[i]+.3])
        action_support[a:b] = True
    low = continuous & (speed>=min_speed) & court_local & ~carried & (state_support|action_support)
    return low,carried


def uncertain_gaps(times, visible, start, end, max_gap=2.):
    """Annotate bounded missing runs without synthesizing coordinates."""
    gaps = []
    i = start
    while i<end:
        if visible[i]:
            i+=1
            continue
        a = i
        while i<end and not visible[i]:
            i+=1
        if a>start and i<end and times[i]-times[a-1]<=max_gap:
            gaps.append({'start_sec':float(times[a]),'end_sec':float(times[i]),
                         'missing_frames':i-a,'coordinates_synthesized':False,
                         'uncertainty':'temporary missing ball evidence inside a candidate; continuity is inferred'})
    return gaps
