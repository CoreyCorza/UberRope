import cv2
import math
import numpy as np
from skimage import transform as trans
import torch
import torchvision
torchvision.disable_beta_transforms_warning()
from torchvision.transforms import v2
from numpy.linalg import norm as l2norm

src1 = np.array([[51.642, 50.115], [57.617, 49.990], [35.740, 69.007],
                 [51.157, 89.050], [57.025, 89.702]],
                dtype=np.float32)
# <--left
src2 = np.array([[45.031, 50.118], [65.568, 50.872], [39.677, 68.111],
                 [45.177, 86.190], [64.246, 86.758]],
                dtype=np.float32)

# ---frontal
src3 = np.array([[39.730, 51.138], [72.270, 51.138], [56.000, 68.493],
                 [42.463, 87.010], [69.537, 87.010]],
                dtype=np.float32)

# -->right
src4 = np.array([[46.845, 50.872], [67.382, 50.118], [72.737, 68.111],
                 [48.167, 86.758], [67.236, 86.190]],
                dtype=np.float32)

# -->right profile
src5 = np.array([[54.796, 49.990], [60.771, 50.115], [76.673, 69.007],
                 [55.388, 89.702], [61.257, 89.050]],
                dtype=np.float32)

src = np.array([src1, src2, src3, src4, src5])
src_map = {112: src, 224: src * 2}

arcface_src = np.array(
    [[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366],
     [41.5493, 92.3655], [70.7299, 92.2041]],
    dtype=np.float32)

arcface_src = np.expand_dims(arcface_src, axis=0)

def pad_image_by_size(img, image_size):
    w, h = math.ceil(img.size(dim=2)), math.ceil(img.size(dim=1))
    if w < image_size or h < image_size:
        # add right, bottom pading to the image if its size is less than image_size value
        add = image_size - min(w, h)
        img = torch.nn.functional.pad(img, (0, add, 0, add), 'constant', 0)

    return img

def transform(img, center, output_size, scale, rotation):
    # pad image by image size
    img = pad_image_by_size(img, output_size)

    scale_ratio = scale
    rot = float(rotation) * np.pi / 180.0
    t1 = trans.SimilarityTransform(scale=scale_ratio)
    cx = center[0] * scale_ratio
    cy = center[1] * scale_ratio
    t2 = trans.SimilarityTransform(translation=(-1 * cx, -1 * cy))
    t3 = trans.SimilarityTransform(rotation=rot)
    t4 = trans.SimilarityTransform(translation=(output_size / 2,
                                                output_size / 2))
    t = t1 + t2 + t3 + t4
    M = t.params[0:2]

    cropped = v2.functional.affine(img, t.rotation, (t.translation[0], t.translation[1]) , t.scale, 0, interpolation=v2.InterpolationMode.BILINEAR, center = (0,0) )
    cropped = v2.functional.crop(cropped, 0,0, output_size, output_size)

    return cropped, M

def trans_points2d(pts, M):
    new_pts = np.zeros(shape=pts.shape, dtype=np.float32)
    for i in range(pts.shape[0]):
        pt = pts[i]
        new_pt = np.array([pt[0], pt[1], 1.], dtype=np.float32)
        new_pt = np.dot(M, new_pt)
        #print('new_pt', new_pt.shape, new_pt)
        new_pts[i] = new_pt[0:2]

    return new_pts

def trans_points3d(pts, M):
    scale = np.sqrt(M[0][0] * M[0][0] + M[0][1] * M[0][1])
    new_pts = np.zeros(shape=pts.shape, dtype=np.float32)
    for i in range(pts.shape[0]):
        pt = pts[i]
        new_pt = np.array([pt[0], pt[1], 1.], dtype=np.float32)
        new_pt = np.dot(M, new_pt)
        #print('new_pt', new_pt.shape, new_pt)
        new_pts[i][0:2] = new_pt[0:2]
        new_pts[i][2] = pts[i][2] * scale

    return new_pts

def trans_points(pts, M):
    if pts.shape[1] == 2:
        return trans_points2d(pts, M)
    else:
        return trans_points3d(pts, M)

def estimate_affine_matrix_3d23d(X, Y):
    ''' Using least-squares solution
    Args:
        X: [n, 3]. 3d points(fixed)
        Y: [n, 3]. corresponding 3d points(moving). Y = PX
    Returns:
        P_Affine: (3, 4). Affine camera matrix (the third row is [0, 0, 0, 1]).
    '''
    X_homo = np.hstack((X, np.ones([X.shape[0],1]))) #n x 4
    P = np.linalg.lstsq(X_homo, Y,rcond=None)[0].T # Affine matrix. 3 x 4
    return P

def P2sRt(P):
    ''' decompositing camera matrix P
    Args:
        P: (3, 4). Affine Camera Matrix.
    Returns:
        s: scale factor.
        R: (3, 3). rotation matrix.
        t: (3,). translation.
    '''
    t = P[:, 3]
    R1 = P[0:1, :3]
    R2 = P[1:2, :3]
    s = (np.linalg.norm(R1) + np.linalg.norm(R2))/2.0
    r1 = R1/np.linalg.norm(R1)
    r2 = R2/np.linalg.norm(R2)
    r3 = np.cross(r1, r2)

    R = np.concatenate((r1, r2, r3), 0)
    return s, R, t

def matrix2angle(R):
    ''' get three Euler angles from Rotation Matrix
    Args:
        R: (3,3). rotation matrix
    Returns:
        x: pitch
        y: yaw
        z: roll
    '''
    sy = math.sqrt(R[0,0] * R[0,0] +  R[1,0] * R[1,0])

    singular = sy < 1e-6

    if  not singular :
        x = math.atan2(R[2,1] , R[2,2])
        y = math.atan2(-R[2,0], sy)
        z = math.atan2(R[1,0], R[0,0])
    else :
        x = math.atan2(-R[1,2], R[1,1])
        y = math.atan2(-R[2,0], sy)
        z = 0

    # rx, ry, rz = np.rad2deg(x), np.rad2deg(y), np.rad2deg(z)
    rx, ry, rz = x*180/np.pi, y*180/np.pi, z*180/np.pi
    return rx, ry, rz

def warp_face_by_bounding_box(img, bboxes, image_size=112):
    # pad image by image size
    img = pad_image_by_size(img, image_size)

    # Set source points from bounding boxes
    source_points = np.array([ [ bboxes[0], bboxes[1] ], [ bboxes[2], bboxes[1] ], [ bboxes[0], bboxes[3] ], [ bboxes[2], bboxes[3] ] ]).astype(np.float32)

    # Set target points from image size
    target_points = np.array([ [ 0, 0 ], [ image_size, 0 ], [ 0, image_size ], [ image_size, image_size ] ]).astype(np.float32)

    # Find transform
    tform = trans.SimilarityTransform()
    tform.estimate(source_points, target_points)
    
    # Transform
    img = v2.functional.affine(img, tform.rotation, (tform.translation[0], tform.translation[1]) , tform.scale, 0, interpolation=v2.InterpolationMode.BILINEAR, center = (0,0) ) 
    img = v2.functional.crop(img, 0,0, image_size, image_size)
    M = tform.params[0:2]

    return img, M

def warp_face_by_face_landmark_5(img, kpss, image_size=112, normalized = False, interpolation=v2.InterpolationMode.BILINEAR, custom_arcface_src = None):
    # pad image by image size
    img = pad_image_by_size(img, image_size)

    M, pose_index = estimate_norm(kpss, image_size, normalized, custom_arcface_src)
    #warped = cv2.warpAffine(img, M, (image_size, image_size), borderValue=0.0)
    t = trans.SimilarityTransform()
    t.params[0:2] = M
    img = v2.functional.affine(img, t.rotation*57.2958, (t.translation[0], t.translation[1]) , t.scale, 0, interpolation=interpolation, center = (0, 0) )
    img = v2.functional.crop(img, 0,0, image_size, image_size)

    return img, M

# lmk is prediction; src is template
def estimate_norm(lmk, image_size=112, normalized = False, custom_arcface_src = None):
    assert lmk.shape == (5, 2)
    tform = trans.SimilarityTransform()
    lmk_tran = np.insert(lmk, 2, values=np.ones(5), axis=1)
    min_M = []
    min_index = []
    min_error = float('inf')

    if custom_arcface_src is None:
        if normalized == False:
            if image_size == 112:
                src = arcface_src
            else:
                src = float(image_size) / 112.0 * arcface_src
        else:
            factor = float(image_size) / 128.0
            src = arcface_src * factor
            src[:, 0] += (factor * 8.0)
    else:
        src = custom_arcface_src

    for i in np.arange(src.shape[0]):
        tform.estimate(lmk, src[i])
        M = tform.params[0:2, :]
        results = np.dot(M, lmk_tran.T)
        results = results.T
        error = np.sum(np.sqrt(np.sum((results - src[i])**2, axis=1)))
        #         print(error)
        if error < min_error:
            min_error = error
            min_M = M
            min_index = i
    return min_M, min_index

def invertAffineTransform(M):
    t = trans.SimilarityTransform()
    t.params[0:2] = M
    IM = t.inverse.params[0:2, :]

    return IM

def warp_face_by_bounding_box_for_landmark_68(img, bbox, input_size):
    """
    :param img: raw image
    :param bbox: the bbox for the face
    :param input_size: tuple input image size
    :return:
    """
    # pad image by image size
    img = pad_image_by_size(img, input_size[0])

    scale = 195 / np.subtract(bbox[2:], bbox[:2]).max()
    translation = (256 - np.add(bbox[2:], bbox[:2]) * scale) * 0.5
    rotation = 0

    t1 = trans.SimilarityTransform(scale=scale)
    t2 = trans.SimilarityTransform(rotation=rotation)
    t3 = trans.SimilarityTransform(translation=translation)

    t = t1 + t2 + t3
    affine_matrix = np.array([ [ scale, 0, translation[0] ], [ 0, scale, translation[1] ] ])

    crop_image = v2.functional.affine(img, t.rotation, (t.translation[0], t.translation[1]) , t.scale, 0, interpolation=v2.InterpolationMode.BILINEAR, center = (0,0) ) 
    crop_image = v2.functional.crop(crop_image, 0,0, input_size[1], input_size[0])

    if torch.mean(crop_image.to(dtype=torch.float32)[0, :, :]) < 30:
        crop_image = cv2.cvtColor(crop_image.permute(1, 2, 0).to('cpu').numpy(), cv2.COLOR_RGB2Lab)
        crop_image[:, :, 0] = cv2.createCLAHE(clipLimit = 2).apply(crop_image[:, :, 0])
        crop_image = torch.from_numpy(cv2.cvtColor(crop_image, cv2.COLOR_Lab2RGB)).to('cuda').permute(2, 0, 1)

    return crop_image, affine_matrix

def warp_face_by_bounding_box_for_landmark_98(img, bbox_org, input_size):
    """
    :param img: raw image
    :param bbox: the bbox for the face
    :param input_size: tuple input image size
    :return:
    """
    # pad image by image size
    img = pad_image_by_size(img, input_size[0])
    
    ##preprocess
    bbox = bbox_org.copy()
    min_face = 20
    base_extend_range = [0.2, 0.3]
    bbox_width = bbox[2] - bbox[0]
    bbox_height = bbox[3] - bbox[1]
    if bbox_width <= min_face or bbox_height <= min_face:
        return None, None
    add = int(max(bbox_width, bbox_height))

    bimg = torch.nn.functional.pad(img, (add, add, add, add), 'constant', 0)

    bbox += add

    face_width = (1 + 2 * base_extend_range[0]) * bbox_width
    center = [(bbox[0] + bbox[2]) // 2, (bbox[1] + bbox[3]) // 2]

    ### make the box as square
    bbox[0] = center[0] - face_width // 2
    bbox[1] = center[1] - face_width // 2
    bbox[2] = center[0] + face_width // 2
    bbox[3] = center[1] + face_width // 2

    # crop
    bbox = bbox.astype(np.int32)
    crop_image = bimg[:, bbox[1]:bbox[3], bbox[0]:bbox[2]]

    h, w = (crop_image.size(dim=1), crop_image.size(dim=2))
    
    t_resize = v2.Resize((input_size[1], input_size[0]), antialias=False)
    crop_image = t_resize(crop_image)

    return crop_image, [h, w, bbox[1], bbox[0], add]

def create_bounding_box_from_face_landmark_106_98_68(face_landmark_106_98_68):
    min_x, min_y = np.min(face_landmark_106_98_68, axis = 0)
    max_x, max_y = np.max(face_landmark_106_98_68, axis = 0)
    bounding_box = np.array([ min_x, min_y, max_x, max_y ]).astype(np.int16)
    return bounding_box

def convert_face_landmark_68_to_5(face_landmark_68, face_landmark_68_score):
    face_landmark_5 = np.array(
    [
        np.mean(face_landmark_68[36:42], axis = 0),
        np.mean(face_landmark_68[42:48], axis = 0),
        face_landmark_68[30],
        face_landmark_68[48],
        face_landmark_68[54]
    ])

    if np.any(face_landmark_68_score):
        face_landmark_5_score = np.array(
        [
            np.mean(face_landmark_68_score[36:42], axis = 0),
            np.mean(face_landmark_68_score[42:48], axis = 0),
            face_landmark_68_score[30],
            face_landmark_68_score[48],
            face_landmark_68_score[54]
        ])
    else:
        face_landmark_5_score = np.array([])

    return face_landmark_5, face_landmark_5_score

def convert_face_landmark_98_to_5(face_landmark_98, face_landmark_98_score):
    face_landmark_5 = np.array(
    [
        face_landmark_98[96], # eye left
        face_landmark_98[97], # eye-right
        face_landmark_98[54], # nose,
        face_landmark_98[76], # lip left
        face_landmark_98[82]  # lip right
    ])
    
    face_landmark_5_score = np.array(
    [
        face_landmark_98_score[96], # eye left
        face_landmark_98_score[97], # eye-right
        face_landmark_98_score[54], # nose,
        face_landmark_98_score[76], # lip left
        face_landmark_98_score[82]  # lip right        
    ])

    return face_landmark_5, face_landmark_5_score

def convert_face_landmark_106_to_5(face_landmark_106):
    face_landmark_5 = np.array(
    [
        face_landmark_106[38], # eye left
        face_landmark_106[88], # eye-right
        face_landmark_106[86], # nose,
        face_landmark_106[52], # lip left
        face_landmark_106[61]  # lip right
    ])

    return face_landmark_5

def convert_face_landmark_478_to_5(face_landmark_478):
    face_landmark_5 = np.array(
    [
        face_landmark_478[468], # eye left
        #np.array([(face_landmark_478[159][0] + face_landmark_478[145][0]) / 2, (face_landmark_478[159][1] + face_landmark_478[145][1]) / 2]), # eye left (145-159)
        face_landmark_478[473], # eye-right
        #np.array([(face_landmark_478[386][0] + face_landmark_478[374][0]) / 2, (face_landmark_478[386][1] + face_landmark_478[374][1]) / 2]), # eye-right (374-386)
        face_landmark_478[4], # nose, 4, 1
        face_landmark_478[61], # lip left ? 61, 57
        face_landmark_478[291]  # lip right ? 291, 287
    ])

    return face_landmark_5

def test_bbox_landmarks(img, bbox, kpss):
        image = img.permute(1,2,0).to('cpu').numpy().copy()
        if len(bbox) > 0:
            box = bbox.astype(int)
            color = (255, 0, 0)
            cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), color, 2)
        
        if len(kpss) > 0:
            for i in range(kpss.shape[0]):
                kps = kpss[i].astype(int)
                color = (0, 0, 255)
                cv2.circle(image, (kps[0], kps[1]), 1, color,
                           2)

        cv2.imshow('image', image) 
        cv2.waitKey(0)         
        cv2.destroyAllWindows()

def test_multi_bbox_landmarks(img, bboxes, kpss):
    if len(bboxes) > 0 and len(kpss) > 0:
        for i in range(np.array(kpss).shape[0]):
            test_bbox_landmarks(img, bboxes[i], kpss[i])

def detect_img_color(img):
    frame = img.permute(1,2,0)

    b = frame[:, :, :1] 
    g = frame[:, :, 1:2] 
    r = frame[:, :, 2:] 
  
    # computing the mean 
    b_mean = torch.mean(b.to(float)) 
    g_mean = torch.mean(g.to(float)) 
    r_mean = torch.mean(r.to(float)) 

    # displaying the most prominent color 
    if (b_mean > g_mean and b_mean > r_mean): 
        return 'BGR'
    elif (g_mean > r_mean and g_mean > b_mean): 
        return 'GBR'

    return 'RGB'