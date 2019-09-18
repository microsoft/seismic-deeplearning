import cv2
import numpy as np
import math
import random


def do_resize2(image, mask, H, W):
    image = cv2.resize(image,dsize=(W,H))
    mask = cv2.resize(mask,dsize=(W,H))
    mask  = (mask>0.5).astype(np.float32)

    return image,mask

def do_horizontal_flip(image):
    #flip left-right
    image = cv2.flip(image,1)
    return image

def do_horizontal_flip2(image,mask):
    image = do_horizontal_flip(image)
    mask  = do_horizontal_flip(mask )
    return image, mask


def do_center_pad(image, pad_left, pad_right):
    return np.pad(image, (pad_left, pad_right), 'edge')

def do_center_pad2(image, mask, pad_left, pad_right):
    image = do_center_pad(image, pad_left, pad_right)
    mask = do_center_pad(mask, pad_left, pad_right)
    return image, mask

def do_invert_intensity(image):
    #flip left-right
    image = np.clip(1-image,0,1)
    return image

def do_brightness_shift(image, alpha=0.125):
    image = image + alpha
    image = np.clip(image, 0, 1)
    return image


def do_brightness_multiply(image, alpha=1):
    image = alpha*image
    image = np.clip(image, 0, 1)
    return image


#https://www.pyimagesearch.com/2015/10/05/opencv-gamma-correction/
def do_gamma(image, gamma=1.0):

    image = image ** (1.0 / gamma)
    image = np.clip(image, 0, 1)
    return image

def do_shift_scale_crop( image, mask, x0=0, y0=0, x1=1, y1=1 ):
    #cv2.BORDER_REFLECT_101
    #cv2.BORDER_CONSTANT

    height, width = image.shape[:2]
    image = image[y0:y1,x0:x1]
    mask  = mask [y0:y1,x0:x1]

    image = cv2.resize(image,dsize=(width,height))
    mask  = cv2.resize(mask,dsize=(width,height))
    mask  = (mask>0.5).astype(np.float32)
    return image, mask


def do_random_shift_scale_crop_pad2(image, mask, limit=0.10):

    H, W = image.shape[:2]

    dy = int(H*limit)
    y0 =   np.random.randint(0,dy)
    y1 = H-np.random.randint(0,dy)

    dx = int(W*limit)
    x0 =   np.random.randint(0,dx)
    x1 = W-np.random.randint(0,dx)

    #y0, y1, x0, x1
    image, mask = do_shift_scale_crop( image, mask, x0, y0, x1, y1 )
    return image, mask

#===========================================================================

def do_shift_scale_rotate2( image, mask, dx=0, dy=0, scale=1, angle=0 ):
    borderMode=cv2.BORDER_REFLECT_101
    #cv2.BORDER_REFLECT_101  cv2.BORDER_CONSTANT

    height, width = image.shape[:2]
    sx = scale
    sy = scale
    cc = math.cos(angle/180*math.pi)*(sx)
    ss = math.sin(angle/180*math.pi)*(sy)
    rotate_matrix = np.array([ [cc,-ss], [ss,cc] ])

    box0 = np.array([ [0,0], [width,0],  [width,height], [0,height], ],np.float32)
    box1 = box0 - np.array([width/2,height/2])
    box1 = np.dot(box1,rotate_matrix.T) + np.array([width/2+dx,height/2+dy])

    box0 = box0.astype(np.float32)
    box1 = box1.astype(np.float32)
    mat  = cv2.getPerspectiveTransform(box0,box1)

    image = cv2.warpPerspective(image, mat, (width,height),flags=cv2.INTER_LINEAR,
                                borderMode=borderMode,borderValue=(0,0,0,))  #cv2.BORDER_CONSTANT, borderValue = (0, 0, 0))  #cv2.BORDER_REFLECT_101
    mask = cv2.warpPerspective(mask, mat, (width,height),flags=cv2.INTER_NEAREST,#cv2.INTER_LINEAR
                                borderMode=borderMode,borderValue=(0,0,0,))  #cv2.BORDER_CONSTANT, borderValue = (0, 0, 0))  #cv2.BORDER_REFLECT_101
    mask  = (mask>0.5).astype(np.float32)
    return image, mask

#https://www.kaggle.com/ori226/data-augmentation-with-elastic-deformations
#https://github.com/letmaik/lensfunpy/blob/master/lensfunpy/util.py
def do_elastic_transform2(image, mask, grid=32, distort=0.2):
    borderMode=cv2.BORDER_REFLECT_101
    height, width = image.shape[:2]

    x_step = int(grid)
    xx = np.zeros(width,np.float32)
    prev = 0
    for x in range(0, width, x_step):
        start = x
        end   = x + x_step
        if end > width:
            end = width
            cur = width
        else:
            cur = prev + x_step*(1+random.uniform(-distort,distort))

        xx[start:end] = np.linspace(prev,cur,end-start)
        prev=cur


    y_step = int(grid)
    yy = np.zeros(height,np.float32)
    prev = 0
    for y in range(0, height, y_step):
        start = y
        end   = y + y_step
        if end > height:
            end = height
            cur = height
        else:
            cur = prev + y_step*(1+random.uniform(-distort,distort))

        yy[start:end] = np.linspace(prev,cur,end-start)
        prev=cur

    #grid
    map_x,map_y =  np.meshgrid(xx, yy)
    map_x = map_x.astype(np.float32)
    map_y = map_y.astype(np.float32)

    #image = map_coordinates(image, coords, order=1, mode='reflect').reshape(shape)
    image = cv2.remap(image, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=borderMode,borderValue=(0,0,0,))


    mask = cv2.remap(mask, map_x, map_y, interpolation=cv2.INTER_NEAREST, borderMode=borderMode,borderValue=(0,0,0,))
    mask  = (mask>0.5).astype(np.float32)
    return image, mask




def do_horizontal_shear2( image, mask, dx=0 ):
    borderMode=cv2.BORDER_REFLECT_101
    #cv2.BORDER_REFLECT_101  cv2.BORDER_CONSTANT

    height, width = image.shape[:2]
    dx = int(dx*width)

    box0 = np.array([ [0,0], [width,0],  [width,height], [0,height], ],np.float32)
    box1 = np.array([ [+dx,0], [width+dx,0],  [width-dx,height], [-dx,height], ],np.float32)

    box0 = box0.astype(np.float32)
    box1 = box1.astype(np.float32)
    mat = cv2.getPerspectiveTransform(box0,box1)

    image = cv2.warpPerspective(image, mat, (width,height),flags=cv2.INTER_LINEAR,
                                borderMode=borderMode,borderValue=(0,0,0,))  #cv2.BORDER_CONSTANT, borderValue = (0, 0, 0))  #cv2.BORDER_REFLECT_101
    mask  = cv2.warpPerspective(mask, mat, (width,height),flags=cv2.INTER_NEAREST,#cv2.INTER_LINEAR
                                borderMode=borderMode,borderValue=(0,0,0,))  #cv2.BORDER_CONSTANT, borderValue = (0, 0, 0))  #cv2.BORDER_REFLECT_101
    mask  = (mask>0.5).astype(np.float32)
    return image, mask