import cv2
import numpy as np
import math
import random
import shapely
import time
from itertools import combinations
import matplotlib.pyplot as plt
import find_table
import itertools

if __name__ == "__main__":
    # img = cv2.imread('images/mytest.jpg') #glare1
    # img = cv2.imread('images/tabletest5.png') #glare2

    # img = cv2.imread('images/tabletest1.png') #basic
    img = cv2.imread('images/easytest.png')  #low light
    img = cv2.resize(img, (1300,700), interpolation=cv2.INTER_AREA)
    original = img.copy()


    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)


    #this works
    # Threshold for "white-ish" — high brightness, low saturation
    # lower = np.array([0, 0, 200])   # Hue doesn't matter, saturation small, value high
    # upper = np.array([179, 70, 255])


    #experiment
    # lower = np.array([0, 0, 200])   # Hue doesn't matter, saturation small, value high
    # upper = np.array([179, 120, 255])

    # mask = cv2.inRange(hsv, lower, upper)
    # white = cv2.bitwise_and(img, img, mask=mask)

    # mask = cv2.blur(mask,(3,3))

    # edges = cv2.Canny(white,100,150)
    # thin_edges = cv2.ximgproc.thinning(edges,thinningType=0)
    # thin_mask = cv2.ximgproc.thinning(mask,thinningType=0)
    # combined_mask = cv2.bitwise_and(thin_edges,mask)
    # combined2 = cv2.bitwise_and(edges,mask)

    #THIS WORKS WHEN THERE IS NO GLARE
    # best_quad,max_area = find_table.find_white_lines_and_largest_contour(combined2,img,maxGap=3,minHoughLine=40,display=True)

    #experiment
    # best_quad,max_area = find_table.find_white_lines_and_largest_contour(thin_mask,img,maxGap=3,display=True)

    # if best_quad:
    #     print(f"area:{max_area}")

    # cv2.imshow("edges",edges)
    # cv2.imshow("white mask",mask)
    # # cv2.imshow("thin mask",thin_mask)
    # # cv2.imshow("thin edges",thin_edges)
    # cv2.imshow("white??",white)
    # cv2.imshow("combined",combined_mask)
    # cv2.imshow("combined2",combined2)
    # cv2.imshow("Largest Contour from Lines", original)

    minHough = [40,15]
    gaps = [3,5,10,15]
    sats = [80,120,170]
    values = [200,160]

    best_quad = None
    max_area = 0

    for combo in itertools.product(gaps,minHough,sats, values):
        gap = combo[0]
        minH = combo[1]
        sat = combo[2]
        val = combo[3]
        lower = np.array([0, 0, val])   # Hue doesn't matter, saturation small, value high
        upper = np.array([179, sat, 255])

        mask = cv2.inRange(hsv, lower, upper)
        white = cv2.bitwise_and(img, img, mask=mask)
        mask = cv2.blur(mask,(3,3))
        edges = cv2.Canny(white,100,150)
        combined2 = cv2.bitwise_and(edges,mask)
        best_quad,max_area = find_table.find_white_lines_and_largest_contour(combined2,img,maxGap=gap,minHoughLine=minH,display=False)
        if best_quad and max_area>20000:
            print(combo)
            break


    if best_quad:
        vert = list(best_quad.exterior.coords)
        vertices = [ (int(round(x)), int(round(y))) for x, y in vert ]
        for point in vertices:
            cv2.circle(original, point, radius=5, color=(0, 255, 0), thickness=-1)  # Green filled circle

        cv2.imshow("?",original)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("STILL NO TABLE FOUND")

    