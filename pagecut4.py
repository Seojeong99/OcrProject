import cv2
import numpy as np


src = cv2.imread('C:\imagedata\document.jpg')
edge = cv2.Canny(src, 50, 150)

contours = cv2.findContours(edge.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[0]
contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]  # 외곽선 크기순으로 정렬

for i in contours:
    length = cv2.arcLength(i, True)
    approx = cv2.approxPolyDP(i, 0.02 * length, True)

    if len(approx) == 4:
        contourCnt = approx
        break

# 투시변환
src_pts = np.array([[approx[0][0][0], approx[0][0][1]],
                    [approx[1][0][0], approx[1][0][1]],
                    [approx[2][0][0], approx[2][0][1]],
                    [approx[3][0][0], approx[3][0][1]]]).astype(np.float32)

w = 300
h = 400

dst_pts = np.array([[0, 0],
                    [0, h - 1],
                    [w - 1, h - 1],
                    [w - 1, 0]]).astype(np.float32)

pers_mat = cv2.getPerspectiveTransform(src_pts, dst_pts)
src_transform = cv2.warpPerspective(src, pers_mat, (w, h))

_, src_binary = cv2.threshold(src_transform, 0, 255, cv2.THRESH_OTSU)
src_filtering = cv2.bilateralFilter(src_binary, -1, 10, 5)

kernel = np.ones((3, 1), np.uint8)
src_morphology = cv2.morphologyEx(src_filtering, cv2.MORPH_OPEN, kernel)