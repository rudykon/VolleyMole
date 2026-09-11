"""Scale/time-normalized movement, with explicit camera and association gates."""
import numpy as np


def camera_transform(previous, current, boxes=()):
    import cv2
    mask = np.full(previous.shape[:2], 255, dtype=np.uint8)
    for x1, y1, x2, y2 in boxes:
        cv2.rectangle(mask, (int(x1), int(y1)), (int(x2), int(y2)), 0, -1)
    points = cv2.goodFeaturesToTrack(previous, 400, .02, 8, mask=mask)
    if points is None or len(points) < 12:
        return None
    moved, status, _ = cv2.calcOpticalFlowPyrLK(previous, current, points, None)
    if moved is None or status is None:
        return None
    keep = status.ravel().astype(bool)
    if keep.sum() < 12:
        return None
    affine, inliers = cv2.estimateAffinePartial2D(points[keep], moved[keep], method=cv2.RANSAC)
    if affine is None or inliers.mean() < .7 or not np.isfinite(affine).all():
        return None
    return affine


def relative_motion(previous, current, dt, camera=None):
    """Match confident boxes one-to-one after global image-motion compensation."""
    unknown = {'body_lengths_per_sec': None, 'pose_lengths_per_sec': None,
               'matched_players': 0, 'camera_compensated': camera is not None}
    if camera is None or not 0 < dt <= 2:
        return unknown
    def valid(rows):
        return [p for p in rows if p.get('confidence', 0) >= .5
                and len(p.get('xyxy', [])) == 4 and p['xyxy'][3] > p['xyxy'][1]]
    before, after = valid(previous), valid(current)
    used, speeds, poses = set(), [], []
    for p in before:
        box = np.asarray(p['xyxy'])
        center = np.array([(box[0]+box[2])/2, (box[1]+box[3])/2, 1.]) @ camera.T
        candidates = []
        for j, q in enumerate(after):
            if j in used: continue
            b = np.asarray(q['xyxy'])
            scale = ((box[3]-box[1])+(b[3]-b[1]))/2
            distance = float(np.linalg.norm(center-(b[:2]+b[2:])/2)/scale)
            if distance < .8 and .6 < (b[3]-b[1])/(box[3]-box[1]) < 1.7:
                candidates.append((distance, j, scale))
        candidates.sort()
        if not candidates or (len(candidates)>1 and candidates[1][0]-candidates[0][0] < .15):
            continue
        distance, j, scale = candidates[0]; used.add(j); speeds.append(distance/dt)
        a, b = np.asarray(p.get('keypoints', [])), np.asarray(after[j].get('keypoints', []))
        if a.ndim == 2 and a.shape == b.shape and a.shape[1] >= 3:
            keep = (a[:, 2] > .5) & (b[:, 2] > .5)
            if keep.sum() >= 4:
                moved = np.c_[a[keep, :2], np.ones(keep.sum())] @ camera.T
                poses.append(float(np.median(np.linalg.norm(b[keep, :2]-moved, axis=1))/scale/dt))
    return {**unknown, 'body_lengths_per_sec': float(np.percentile(speeds, 75)) if speeds else None,
        'pose_lengths_per_sec': float(np.percentile(poses, 75)) if poses else None, 'matched_players': len(speeds)}


def ball_motion(previous, current, dt, camera, players):
    """No trajectory apex/contact equivalence; fast ball details need dense PTS."""
    result = {'body_lengths_per_sec': None, 'relative_velocity': None, 'low_ball': None,
        'near_wrist': None, 'confirmed_touch': None, 'net_crossing': None, 'landing': None}
    if previous is None or current is None or camera is None or not 0 < dt <= .2:
        return result
    people = [p for p in players if p.get('confidence', 0) >= .5 and p['xyxy'][3] > p['xyxy'][1]]
    if not people: return result
    xy = np.asarray(current)
    # Use the nearest visible person's scale; do not count all detector boxes.
    person = min(people, key=lambda p: np.linalg.norm((np.array(p['xyxy'][:2])+p['xyxy'][2:])/2-xy))
    box = person['xyxy']; scale = box[3]-box[1]
    velocity = (xy-np.r_[previous, 1.] @ camera.T)/dt/scale
    result.update(body_lengths_per_sec=float(np.linalg.norm(velocity)), relative_velocity=velocity.tolist(),
        low_ball=bool(xy[1] > box[1]+.75*scale))
    kp = person.get('keypoints', [])
    wrists = [np.asarray(kp[i][:2]) for i in (9,10) if len(kp)>i and len(kp[i])>2 and kp[i][2]>.5]
    if wrists: result['near_wrist'] = bool(min(np.linalg.norm(xy-w) for w in wrists)/scale < .2)
    return result
