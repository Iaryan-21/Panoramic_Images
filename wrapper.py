import numpy as np 
import cv2 
import matplotlib.pyplot as plt 
import math 
import numpy.linalg 
import glob 
import copy 
import skimage.feature 
from skimage.feature import peak_local_max 
import argparse 
import os 
import random 

def anms(cmap):
    dist = 0
    maxima = peak_local_max(cmap, min_distance=10)
    
    nbest = 800
    nstrong = maxima.shape[0]
    r = np.Infinity * np.ones([nstrong,3])
    for i in range(nstrong):
        for j in range(nstrong):
            xi = maxima[i][1]
            xj = maxima[j][1]
            yi = maxima[i][0]
            yj = maxima[j][0]
            if(cmap[yj,xj] > cmap[yi,xi]):
                dist = np.square(xj -xi) + np.square(yj-yi)
            if dist < r[i,0]:
                r[i,0] = dist
                r[i,1] = xi
                r[i,2] = yi
    corners = r[np.argsort(-r[:, 0])]
    
    best_corners = corners[:nbest,:]
    return best_corners

def corner_det(img1,img2):
    i_1 = img1.copy()
    i_2 = img2.copy()
    mask_1 = np.full((img1.shape[0],img1.shape[1]),0,dtype=np.uint8)
    mask_2 = np.full((img2.shape[0],img2.shape[1]),0,dtype=np.uint8)
    gray_1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray_2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    corners_1 = cv2.goodFeaturesToTrack(gray_1, 10000, 0.001,10)
    for i in corners_1:
        x,y = i.ravel()
        cv2.circle(i_1,(int(x),int(y)),2,(255,0,0),-1)
        cv2.circle(mask_1,(int(x),int(y)),5,(255,255,255),-1)
    corner_img1 = cv2.bitwise_and(gray_1,mask_1)

    corners_2 = cv2.goodFeaturesToTrack(gray_2, 10000,0.001,10)
    for i in corners_2:
        x,y = i.ravel()
        cv2.circle(i_2,(int(x),int(y)),2,(255,0,0),-1)
        cv2.circle(mask_2,(int(x),int(y)),5,(255,255,255),-1)
    corner_img2 = cv2.bitwise_and(gray_2,mask_2)
    
    return i_1,i_2,corner_img1, corner_img2


def image_disp(img_list):
    #taking minimum heights
    h_min = min(img.shape[0] for img in img_list)

    combined = [cv2.resize(img,(int(img.shape[1] *h_min/ img.shape[0]),h_min),interpolation =cv2.INTER_CUBIC )
                for img in img_list]
    final = cv2.hconcat(combined)
    cv2.imshow('input',final)
    cv2.waitKey()
    cv2.destroyAllWindows()

    return final

def feature_descriptor(img, corner):
    patch_size = 40
    img = np.pad(img, (patch_size,), 'constant', constant_values=0)
    features = np.zeros((64, 1))
    for i in range(corner.shape[0]):
        x, y = corner[i][1], corner[i][2]
        # Skip if corner coordinates are infinite
        if np.isinf(x) or np.isinf(y):
            continue
        patch = img[int(y + (patch_size / 2)):int(y + (3 * patch_size / 2)),int(x + (patch_size / 2)):int(x + (3 * patch_size / 2))]
        patch_blur = cv2.GaussianBlur(patch, (5, 5), 0)
        patch_blur = cv2.resize(patch_blur, (8, 8), interpolation=cv2.INTER_AREA)
        feature = patch_blur.reshape((64, 1))
        feature = (feature - np.mean(feature)) / (np.std(feature) + 0.0000001)
        features = np.dstack((features, feature))
    return features[:, :, 1:]

def feature_matcher(features1, features2, corners_1, corners_2):
    _,x,a = features1.shape
    _,y,b = features2.shape
    a = int(min(a,b))
    b = int(max(a,b))
    matched = []

    for i in range(a):
        match = {}
        for j in range(b):
            ssd = np.linalg.norm((features1[:,:,i]-features2[:,:,j]))**2
            match[ssd] = [corners_1[i,:],corners_2[j,:]]
        ratio = sorted(match)
        if ratio[0]/ratio[1] < 0.8:
            pairs = match[ratio[0]]
            matched.append(pairs)
    
    return matched


def resizing(imgs):
    images = imgs.copy()
    sizes = []
    resized = []
    for image in images:
        x,y,ch = image.shape
        sizes.append([x,y,ch])

    sizes = np.array(sizes)
    x_target, y_target, _ = np.max(sizes, axis = 0)

    for i, image in enumerate(images):
        resize = np.zeros((x_target, y_target, sizes[i,2]), np.uint8)
        resize[0:sizes[i,0], 0:sizes[i,1], 0:sizes[i,2]] = image
        resized.append(resize)

    return resized

def display_featurematch(img1,img2,matchPairs):
 
    pt1 = []
    pt2 = []
    img1, img2 = resizing([img1, img2])
    new = np.concatenate((img1, img2), axis = 1)
    for i in range(len(matchPairs)):
        x1 = int(matchPairs[i][0][1])
        y1 = int(matchPairs[i][0][2])
        x2 = int(matchPairs[i][1][1])
        y2 = int(matchPairs[i][1][2])
        pt1.append([x1,y1])
        pt2.append([x2,y2])
        
        cv2.line(new,(x1,y1),(x2+int(img1.shape[1]),y2),(255, 0, 0), 1)
        cv2.circle(new,(x1,y1),1,(255,255,0),-1)
        cv2.circle(new,(x2+int(img1.shape[1]),y2),1,(255,255,0),-1)

    cv2.imshow("feature matching",new)
    cv2.imwrite('feature_match.png',new)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return pt1,pt2

def RANSAC(pt1,pt2, thresh):

    Hdash = np.zeros([3, 3])
    outliers =25
    accuracy = 0.9
    e = outliers / pt1.shape[0]
    s = 4
    p = 0.9
    no_iter = np.log(1 - p) / np.log(1 - np.power((1 - e), s)) 
    no_iter = 6000
    N_best = 0
    final = []
    

    for i in range(int(no_iter)):

        index = np.random.choice(pt1.shape[0], size=4)

        points1 = pt1[index]
        points2 = pt2[index]
    
        H = cv2.getPerspectiveTransform(np.float32(points1), np.float32(points2))
        
        pt1_dash = np.vstack((pt1[:,0], pt1[:,1], np.ones([1, pt1.shape[0]])))
        point1_dash = np.dot(H, pt1_dash)
        
        a = point1_dash[0,:]/(point1_dash[2,:] + 0.0000001)
        b = point1_dash[1,:]/(point1_dash[2,:] + 0.0000001)
        p1_dash = np.array([a, b]).T
        diff = pt2 - p1_dash
        error = np.zeros(pt2.shape[0])
        for n in range(pt2.shape[0]):
            
            error[n] = np.linalg.norm(diff[n])

        error[error <= thresh] = 1
        error[error > thresh] = 0
        current = np.sum(error)

        if current > N_best:
            N_best = current
            H_dash = H
            final = np.where(error == 1)
    
    img_point1 =  pt1[final]
    img_point2 =  pt2[final]
    pairs = np.zeros([img_point1.shape[0], img_point1.shape[1], 2])
    pairs[:, 0, :] = img_point1
    pairs[:, 1, :] = img_point2


    print("RANSAC filtered:", len(pairs))
    return H_dash, pairs.astype(int)

def display_ransac(img1, img2, match):
    image_1 = img1.copy()
    image_2 = img2.copy()
    image_1, image_2 = resizing([image_1, image_2])
    combine = np.concatenate((image_1, image_2), axis = 1)
    corners_1 = match[:,0].copy()
    corners_2  = match[:,1].copy()
    corners_2[:,0] += image_1.shape[1]
    for a,b in zip(corners_1, corners_2):
        cv2.line(combine, (int(a[0]),int(a[1])), (int(b[0]),int(b[1])), (255, 0, 0), 1)
        cv2.circle(combine,(int(a[0]),int(a[1])),1,(255,255,0),-1)
        cv2.circle(combine,(int(b[0]),int(b[1])),1,(255,255,0),-1)
    cv2.imshow("ransac", combine)
    cv2.imwrite('ransac.png',combine)
    cv2.waitKey() 
    cv2.destroyAllWindows()

def Blend(img1, img2, H):
    merge = []
    img1, img2 = resizing([img1, img2])
    h1,w1 = img1.shape[:2]
    h2,w2 = img2.shape[:2]

    pt1 = np.float32([[0, 0], [0, h1], [w1, h1], [w1, 0]]).reshape(-1,1,2)
    pt2 = np.float32([[0, 0], [0, h2], [w2, h2], [w2, 0]]).reshape(-1,1,2)

    pt1_dash = cv2.perspectiveTransform(pt1, H)

    pts = np.concatenate((pt1_dash, pt2), axis = 0)
    
    #find min and max for new image
    for p in range(len(pts)):
        merge.append(pts[p].ravel())

    x_min, y_min = np.int0(np.min(np.array(merge), axis = 0))
    x_max, y_max = np.int0(np.max(np.array(merge), axis = 0))
    Hdash = np.array([[1, 0, -x_min], [0, 1, -y_min], [0, 0, 1]]) # translate
    stitch1 = cv2.warpPerspective(img1, np.dot(Hdash, H), (x_max-x_min, y_max-y_min))
    blend = stitch1.copy()
    blend[-y_min:-y_min+h1, -x_min: -x_min+w1] = img2

    r = np.where(img2 == [0,0,0])
    y = r[0] + -y_min 
    x = r[1] + -x_min 

    blend[y,x] = stitch1[y,x]
    
    return blend

def panorama(image1, image2):
    image1, image2 = resizing([image1, image2])
    c1, c2 = [],[]
    img1 = image1.copy()
    img2 = image2.copy()
    gray1 = cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)
    print("Detecting Corners")
    i_1, i_2, corner_img1, corner_img2 = corner_det(image1, image2)
    cv2.imwrite('corners1.jpg',i_1)
    cv2.imwrite('corners2.jpg',i_2)
    print("Performing Adaptive Non-Maximal Suppression")
    c1 = anms(corner_img1)
    for corner1 in c1:
        _,x1,y1 = corner1.ravel()
        if not math.isinf(x1) and not math.isinf(y1):
            cv2.circle(img1, (int(x1), int(y1)), 1, (255, 0, 0), -1)
    cv2.imwrite('anms_1.png',img1)

    c2 = anms(corner_img2)
    for corner2 in c2:
        _,x2,y2 = corner2.ravel()
        if not math.isinf(x2) and not math.isinf(y2):
            cv2.circle(img2, (int(x2), int(y2)), 1, (255, 0, 0), -1)
    cv2.imwrite('anms_2.png',img2)

    f1 = feature_descriptor(gray1,c1)
    f2 = feature_descriptor(gray2,c2)
    print("Feature Matching")
    img1 = image1.copy()
    img2 = image2.copy()
    match = feature_matcher(f1,f2,c1,c2)
    print("Number of matches",len(match))
    pt1,pt2 = display_featurematch(img1,img2,match)

    print("RANSAC")
    im1 = image1.copy()
    im2 = image2.copy()
    H,pairs = RANSAC(np.array(pt1),np.array(pt2),5)
    display_ransac(img1,img2,pairs)

    return H

def main():
    Parser = argparse.ArgumentParser()
    Parser.add_argument('--Folder', default='C:\\Users\\aryan\\OneDrive\\Desktop\\panorama-stitching\\Data\\Train\\CustomSet1\\')
    Args = Parser.parse_args()
    path = Args.Folder

    images = [cv2.imread(file)for file in sorted(glob.glob(str(path)+'/*.jpg'))]
    print("Total Number of images are:", len(images))
    print("Displaying images")
    image_disp(images)
    N = len(images)
    image1 = images[0]
    for image2 in images[1:]:

        H = panorama(image1,image2)
        final= Blend(image1,image2, H)
        
        cv2.imshow('Panorama', final)
        cv2.imwrite('mypano.png',final)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        image1 = final

if __name__ == '__main__':
    main()