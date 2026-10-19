"""Conservative exclusion regions; detector boxes and manual glazing polygons."""
import numpy as np


def person_boxes(boxes, labels, scores, image_size, threshold=.3, padding=.1):
    width, height = image_size
    output = []
    for box, label, score in zip(boxes, labels, scores, strict=True):
        if int(label) != 1 or score < threshold: continue
        x0, y0, x1, y1 = map(float, box)
        if not np.isfinite([x0, y0, x1, y1, score]).all() or x1 <= x0 or y1 <= y0:
            raise ValueError('Invalid detector person box')
        dx, dy = max(2, padding*(x1-x0)), max(2, padding*(y1-y0))
        output.append([max(0, (x0-dx)/width), max(0, (y0-dy)/height),
                       min(1, (x1+dx)/width), min(1, (y1+dy)/height)])
    return output


def annotation_mask(shape, entry):
    import cv2
    cv2.setNumThreads(1)
    h, w = shape
    if min(h, w) <= 0: raise ValueError('Mask grid must be positive')
    people, glass = np.zeros(shape, bool), np.zeros(shape, np.uint8)
    for box in entry.get('people_boxes', []):
        if (len(box) != 4 or not np.isfinite(box).all() or min(box) < 0 or max(box) > 1
                or box[2] <= box[0] or box[3] <= box[1]):
            raise ValueError('Invalid normalized person box')
        x0, y0 = np.floor(np.array(box[:2])*[w, h]).astype(int)
        x1, y1 = np.ceil(np.array(box[2:])*[w, h]).astype(int)
        people[y0:y1, x0:x1] = True
    for polygon in entry.get('glass_polygons', []):
        xy = np.asarray(polygon, float)
        if xy.ndim != 2 or xy.shape[1] != 2 or len(xy) < 3 or not np.isfinite(xy).all() or (xy < 0).any() or (xy > 1).any():
            raise ValueError('Invalid normalized glazing polygon')
        points = np.rint(xy*[w-1, h-1]).astype(np.int32)
        cv2.fillPoly(glass, [points], 1)
    glass = glass.astype(bool)
    return {'people': people, 'glass': glass, 'excluded': people | glass}
