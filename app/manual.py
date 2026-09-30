"""Deterministic manual adjustment, mirrored by the browser live preview."""
import math
import cv2
import numpy as np
from .straighten import StraightenResult


def manual_geometry(width, height, rotation, vertical, horizontal):
    values = (rotation, vertical, horizontal)
    if not all(math.isfinite(x) for x in values) or abs(rotation) > 15 or abs(vertical) > 100 or abs(horizontal) > 100:
        raise ValueError('Invalid manual adjustment')
    if width < 32 or height < 32:
        raise ValueError('Photo is too small')
    rad = math.radians(rotation)
    c, s = math.cos(rad), math.sin(rad)
    r = np.array([[c,-s,0],[s,c,0],[0,0,1]], dtype=np.float64)
    p = np.array([[1,0,0],[0,1,0],[horizontal*.005/width,vertical*.005/height,1]], dtype=np.float64)
    to_origin = np.array([[1,0,-width/2],[0,1,-height/2],[0,0,1]], dtype=np.float64)
    matrix = np.linalg.inv(to_origin) @ p @ r @ to_origin
    inverse = np.linalg.inv(matrix)
    def safe(scale):
        for x in [-1,1]:
            for y in [-1,1]:
                point = inverse @ [width/2+x*width*scale/2,height/2+y*height*scale/2,1]
                if point[2] <= 0:
                    return False
                sx, sy = point[:2]/point[2]
                if sx < 5 or sy < 5 or sx > width-6 or sy > height-6:
                    return False
        return True
    lo, hi = 0., 1.
    for _ in range(45):
        mid = (lo+hi)/2
        if safe(mid): lo = mid
        else: hi = mid
    divisor = math.gcd(width,height)
    uw, uh = width//divisor, height//divisor
    n = math.floor(min(width*lo/uw,height*lo/uh))
    if n < 1:
        raise ValueError('No safe crop at the original aspect ratio')
    w, h = n*uw, n*uh
    left, top = (width-w)/2, (height-h)/2
    translation = np.array([[1,0,-left],[0,1,-top],[0,0,1]], dtype=np.float64)
    return translation @ matrix, w, h


def adjust(image, rotation, vertical, horizontal):
    if rotation == vertical == horizontal == 0:
        return StraightenResult(image.copy(),0,False,1,0,'none',{'warnings':[]})
    height, width = image.shape[:2]
    matrix, w, h = manual_geometry(width,height,rotation,vertical,horizontal)
    output = cv2.warpPerspective(image,matrix,(w,h),flags=cv2.INTER_LANCZOS4,borderMode=cv2.BORDER_CONSTANT)
    loss = 1-w*h/(width*height)
    return StraightenResult(output,-rotation,bool(vertical or horizontal),1,loss,'manual',
        {'warnings':[], 'manual':dict(rotation=rotation,vertical=vertical,horizontal=horizontal),
         'effective_homography':matrix.tolist()})
