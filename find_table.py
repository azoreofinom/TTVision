import cv2
import numpy as np
import math
import random
import shapely
import time
from itertools import combinations,permutations
import matplotlib.pyplot as plt

LINE_CONNECTION_DIST = 100 #max distance between the endpoints of 2 lines which should be connected. this greatly influences the number of potential lines
MIN_LINE_LENGTH = 50 #the minimum length of a line segment for it to be considered a potential side of the table (before extension)
MAX_LINES_TO_PROCESS = 80 #finding all the 4 segment combinations from more than 80 lines takes forever

class HoughBundler:     
    def __init__(self,min_distance=5,min_angle=2):
        self.min_distance = min_distance
        self.min_angle = min_angle

    def get_orientation(self,line):
        orientation = math.atan2((line[3] - line[1]), (line[2] - line[0]))
        return (math.degrees(orientation) % 180)
    
    def are_collinear(self,l1,l2):
        A = (l1[0],l1[1])
        B = (l1[2],l1[3])
        C = (l2[0],l2[1])
        D = (l2[2],l2[3])
        AC = [l1[0], l1[1], l2[0], l2[1]]
        BD = [l1[2], l1[3], l2[2], l2[3]]

        l1_angle = self.get_orientation(l1)
        l2_angle = self.get_orientation(l2)
        ac_angle = self.get_orientation(AC)
        bd_angle = self.get_orientation(BD)
        angle_diff1 = abs(l1_angle-ac_angle)%180
        angle_diff1 = min(angle_diff1,180-angle_diff1)

        angle_diff2 = abs(l1_angle-bd_angle)%180
        angle_diff2 = min(angle_diff2,180-angle_diff2)

        if ((angle_diff1>5)and(math.dist(A,C)>2)) or ((angle_diff2>5)and math.dist(B,D)>2):
            return False
        return True


    # def get_orientation(self, line): #values 0-90
    #     orientation = math.atan2(abs((line[3] - line[1])), abs((line[2] - line[0])))
    #     return math.degrees(orientation)
    
    

    def check_is_line_different(self, line_1, groups, min_distance_to_merge, min_angle_to_merge):
        for group in groups:
            for line_2 in group:
                if self.get_distance(line_2, line_1) < min_distance_to_merge:
                    orientation_1 = self.get_orientation(line_1)
                    orientation_2 = self.get_orientation(line_2)
                    if abs(orientation_1 - orientation_2) < min_angle_to_merge and self.are_collinear(line_2,line_1):
                        group.append(line_1)
                        return False
        return True

    def distance_point_to_line(self, point, line):
        px, py = point
        x1, y1, x2, y2 = line

        def line_magnitude(x1, y1, x2, y2):
            line_magnitude = math.sqrt(math.pow((x2 - x1), 2) + math.pow((y2 - y1), 2))
            return line_magnitude

        lmag = line_magnitude(x1, y1, x2, y2)
        if lmag < 0.00000001:
            distance_point_to_line = 9999
            return distance_point_to_line

        u1 = (((px - x1) * (x2 - x1)) + ((py - y1) * (y2 - y1)))
        u = u1 / (lmag * lmag)

        if (u < 0.00001) or (u > 1):
            #// closest point does not fall within the line segment, take the shorter distance
            #// to an endpoint
            ix = line_magnitude(px, py, x1, y1)
            iy = line_magnitude(px, py, x2, y2)
            if ix > iy:
                distance_point_to_line = iy
            else:
                distance_point_to_line = ix
        else:
            # Intersecting point is on the line, use the formula
            ix = x1 + u * (x2 - x1)
            iy = y1 + u * (y2 - y1)
            distance_point_to_line = line_magnitude(px, py, ix, iy)

        return distance_point_to_line

    def get_distance(self, a_line, b_line):
        dist1 = self.distance_point_to_line(a_line[:2], b_line)
        dist2 = self.distance_point_to_line(a_line[2:], b_line)
        dist3 = self.distance_point_to_line(b_line[:2], a_line)
        dist4 = self.distance_point_to_line(b_line[2:], a_line)

        return min(dist1, dist2, dist3, dist4)

    def merge_lines_into_groups(self, lines):
        groups = []  # all lines groups are here
        # first line will create new group every time
        groups.append([lines[0]])
        # if line is different from existing gropus, create a new group
        for line_new in lines[1:]:
            if self.check_is_line_different(line_new, groups, self.min_distance, self.min_angle):
                groups.append([line_new])

        return groups

    def merge_line_segments(self, lines):
        orientation = self.get_orientation(lines[0])
      
        if(len(lines) == 1):
            return np.block([[lines[0][:2], lines[0][2:]]])

        points = []
        for line in lines:
            points.append(line[:2])
            points.append(line[2:])
        if 45 < orientation <= 90:
            #sort by y
            points = sorted(points, key=lambda point: point[1])
        else:
            #sort by x
            points = sorted(points, key=lambda point: point[0])

        return np.block([[points[0],points[-1]]])

    def process_lines(self, lines):
        lines_horizontal  = []
        lines_vertical  = []
  
        for line_i in [l[0] for l in lines]:
            orientation = self.get_orientation(line_i)
            # if vertical
            if 45 < orientation <= 90:
                lines_vertical.append(line_i)
            else:
                lines_horizontal.append(line_i)

        lines_vertical  = sorted(lines_vertical , key=lambda line: line[1])
        lines_horizontal  = sorted(lines_horizontal , key=lambda line: line[0])
        merged_lines_all = []

        # for each cluster in vertical and horizantal lines leave only one line
        for i in [lines_horizontal, lines_vertical]:
            if len(i) > 0:
                groups = self.merge_lines_into_groups(i)
                merged_lines = []
                for group in groups:
                    merged_lines.append(self.merge_line_segments(group))
                merged_lines_all.extend(merged_lines)
                    
        return np.asarray(merged_lines_all)
    



def get_intersection(p1, p2, p3, p4):
    #not 100% sure about this, maybe reimplement with line segment formulas
    """Returns the intersection point of lines p1-p2 and p3-p4, or None if they don't intersect."""
    x1, y1, x2, y2 = *p1, *p2
    x3, y3, x4, y4 = *p3, *p4

    denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if denom == 0:
        return None  # Parallel lines

    px = ((x1 * y2 - y1 * x2)*(x3 - x4) - (x1 - x2)*(x3 * y4 - y3 * x4)) / denom
    py = ((x1 * y2 - y1 * x2)*(y3 - y4) - (y1 - y2)*(x3 * y4 - y3 * x4)) / denom

    # Check if intersection is within the line segments
    if (
        min(x1, x2) - 1e-5 <= px <= max(x1, x2) + 1e-5 and
        min(y1, y2) - 1e-5 <= py <= max(y1, y2) + 1e-5 and
        min(x3, x4) - 1e-5 <= px <= max(x3, x4) + 1e-5 and
        min(y3, y4) - 1e-5 <= py <= max(y3, y4) + 1e-5
    ):
        return int(px), int(py)
    return None


def get_line_intersection(p1,p2,p3,p4):
    #get intersection point, even if the segments are not currently interesecting
    """Returns the intersection point of lines p1-p2 and p3-p4, or None if they don't intersect."""
    x1, y1, x2, y2 = *p1, *p2
    x3, y3, x4, y4 = *p3, *p4

    denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if denom == 0:
        return None  # Parallel lines

    px = ((x1 * y2 - y1 * x2)*(x3 - x4) - (x1 - x2)*(x3 * y4 - y3 * x4)) / denom
    py = ((x1 * y2 - y1 * x2)*(y3 - y4) - (y1 - y2)*(x3 * y4 - y3 * x4)) / denom

    return int(px), int(py)



def get_existing_inter(line_list):
    existing_inter = []
    for i, (x1, y1, x2, y2) in enumerate(line_list):
        res = []
        p1, p2 = (x1, y1), (x2, y2)
        for j, (x3, y3, x4, y4) in enumerate(line_list):
            if i == j:
                continue
            
            p3, p4 = (x3, y3), (x4, y4)
            inter = get_intersection(p1, p2, p3, p4)
            if inter:
                res.append(j)
        
        existing_inter.append(res)
    return existing_inter


def get_extension_side(p1,p2,p3,p4):
    pairs = [(p1,p3),(p1,p4),(p2,p3),(p2,p4)]
    # Find the smallest distance and the corresponding pair
    min_dist = float('inf')
    min_pair = None

    for a, b in pairs:
        dist = math.dist(a, b)
        if dist < min_dist:
            min_dist = dist
            min_pair = (a, b)
    
    intersection_point = get_line_intersection(p1,p2,p3,p4)
    if (intersection_point and min_dist<LINE_CONNECTION_DIST):
        k = pairs.index(min_pair)
        if (k==0 or k==1) and math.dist(p1,intersection_point) < (LINE_CONNECTION_DIST*2): #and (math.dist(intersection_point,p2)>math.dist(p1,p2)):
            # new_p1 = intersection_point
            return (1,intersection_point)
        elif (k==2 or k==3) and math.dist(p2,intersection_point) < (LINE_CONNECTION_DIST*2): #and (math.dist(p1,intersection_point)>math.dist(p1,p2)):
            # new_p2 = intersection_point
            return (2,intersection_point)
        else:
            return None
    else:
        return None

# def get_connected_lines(lines):
#     final_lines = []    
#     # lines = [tuple(line[0]) for line in lines]
#     existing = get_existing_inter(lines)
 
#     # print(f"different format:{lines[0]}")
#     for i, (x1, y1, x2, y2) in enumerate(lines):
#         p1, p2 = (x1, y1), (x2, y2)
#         final_lines.append((*p1,*p2)) 
#         for j, (x3, y3, x4, y4) in enumerate(lines):
#             new_p1 = p1
#             new_p2 = p2
#             if i == j or (j in existing[i]):
#                 continue
#             p3, p4 = (x3, y3), (x4, y4)

#             pairs = [(p1,p3),(p1,p4),(p2,p3),(p2,p4)]
#             # Find the smallest distance and the corresponding pair
#             min_dist = float('inf')
#             min_pair = None

#             for a, b in pairs:
#                 dist = math.dist(a, b)
#                 if dist < min_dist:
#                     min_dist = dist
#                     min_pair = (a, b)
            
#             intersection_point = get_line_intersection(p1,p2,p3,p4)
#             if (intersection_point and min_dist<LINE_CONNECTION_DIST):
#                 k = pairs.index(min_pair)
#                 if (k==0 or k==1) and math.dist(p1,intersection_point) < (LINE_CONNECTION_DIST*2):
#                     new_p1 = intersection_point
#                 elif (k==2 or k==3) and math.dist(p2,intersection_point) < (LINE_CONNECTION_DIST*2):
#                     new_p2 = intersection_point
#                 final_lines.append((*new_p1,*new_p2))    #INDENTATION! THIS WAY MULTIPLE PAIRS OF LINES ARE CONNECTED       

        
        
#         # final_lines.append((*new_p1,*new_p2))    #INDENTATION! THIS WAY MULTIPLE PAIRS OF LINES ARE CONNECTED       
                    
#     return final_lines

def get_connected_lines2(lines):
    final_lines = []
    #permutations() ensures lines are not repeated(ie try to connect with themselves)
    for combo in permutations(lines, 3): 
        p1, p2 = (combo[0][0],combo[0][1]), (combo[0][2],combo[0][3])
        p3, p4 = (combo[1][0],combo[1][1]), (combo[1][2],combo[1][3])
        p5, p6 = (combo[2][0],combo[2][1]), (combo[2][2],combo[2][3])
        
        inter1 = get_extension_side(p1,p2,p3,p4)
        inter2 = get_extension_side(p1,p2,p5,p6)
        # print(inter1)
        # print(inter2)

        if inter1 and inter2 and inter1[0]==1 and inter2[0]==2:
            # print('asd')
            # print((p1,p2))
            # print((p3,p4))
            # print((p5,p6))
            # print((inter1[1],inter2[1]))
            final_lines.append((*inter1[1],*inter2[1]))
    
    return final_lines
         




def get_largest_quad(line_list):
    print(f"nr of lines:{len(line_list)}")
    if len(line_list)>MAX_LINES_TO_PROCESS:
        return None,None
    
    lines =  [ [(x1, y1), (x2, y2)] for x1, y1, x2, y2 in line_list ]
    # Convert each line into a Shapely LineString
    line_strings = [shapely.LineString(line) for line in lines]

    max_area = 0
    best_quad = None
    for combo in combinations(line_strings, 4):
        # print(combo)   
        # Use polygonize to extract all closed polygons formed by lines
        polygons = shapely.polygonize(combo)
        if not polygons:
            continue
        
        
        for geom in polygons.geoms:
            poly = shapely.Polygon(geom)
            area = poly.area
            print(f"area?????{area}")
            if (len(geom.exterior.coords) - 1 == 4) and shapely.is_simple(geom) and area > max_area:
                max_area = area
                best_quad = poly
    

    # Step 4: Output result
    if best_quad:
        print("Max Area:", max_area)
        print("Vertices:", list(best_quad.exterior.coords))
    else:
        print("No simple quadrilateral found.")
        return None,None

  
    return best_quad,max_area




def GW_white_balance(img):
    img_LAB = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    avg_a = np.average(img_LAB[:, :, 1])
    avg_b = np.average(img_LAB[:, :, 2])
    img_LAB[:, :, 1] = img_LAB[:, :, 1] - ((avg_a - 128) * (img_LAB[:, :, 0] / 255.0) * 3.2)
    img_LAB[:, :, 2] = img_LAB[:, :, 2] - ((avg_b - 128) * (img_LAB[:, :, 0] / 255.0) * 3.2)
    balanced_image = cv2.cvtColor(img_LAB, cv2.COLOR_LAB2BGR)
    return balanced_image

# Function to stretch histogram channel-wise
def contrast_stretch(channel):
    min_val = np.min(channel)
    max_val = np.max(channel)
    stretched = 255 * (channel - min_val) / (max_val - min_val)
    return stretched.astype(np.uint8)

def find_white_lines_and_largest_contour(white_mask,original,maxGap=3,minHoughLine=15,display=False):
    # ,(946,345,939,360)

    # testLines = [(338,399,205,476),(218,478,930,489),(879,413,965,487),(420,397,804,406), (403,349,403,365),(946,345,939,360),(946,489,962,488),(603,384,595,399),(970,399,990,395),(833,513,833,528),(536,513,551,513)] #last element is the extra line
    # (90, 592, 109, 590)

    # testLines = [(90, 592, 109, 590),(206, 477, 340, 398), (218, 478, 930, 489), (301, 333, 319, 333), (331, 335, 347, 334), 
    #              (400, 339, 415, 335), (421, 398, 804, 406), (536, 513, 551, 513), (583, 337, 601, 338), 
    #              (595, 402, 603, 383), (881, 414, 966, 489), (935, 344, 998, 344), (939, 362, 948, 345), 
    #              (946, 489, 964, 488), (970, 400, 992, 395), (404, 348, 404, 366), (834, 529, 834, 512)]
    

    # # testLines = [(206, 477, 340, 398), (218, 478, 930, 489), (881, 414, 966, 489), (421, 398, 804, 406)]
    # #left,bottom,right,top

    # # testLines = [(218, 478, 930, 489), (881, 414, 966, 489), (421, 398, 804, 406)]
    # # print(testLines)
    # lines = get_connected_lines2(testLines)
    # # print("TEST!!!!")
    # # print(lines)

    # basic_lines = np.zeros_like(original)
    # start = time.time()
    # vertices = get_largest_quad(lines)
    # print(vertices)
    # end = time.time()
    # print(f"{end - start} seconds")
    # # Draw points
    # if not vertices:
    #     print("TABLE NOT FOUND")
    # else:
    #     for point in vertices:
    #         cv2.circle(basic_lines, point, radius=5, color=(0, 255, 0), thickness=-1)  # Green filled circle

    
    # if lines is not None:
    #     for line in lines:
    #         x1, y1, x2, y2 = line
    #         # print(line)
    #         # Generate a random RGB color
    #         r = random.randint(0, 255)
    #         g = random.randint(0, 255)
    #         b = random.randint(0, 255)
    #         # Convert RGB to BGR for OpenCV
    #         color = (b, g, r)
    #         cv2.line(basic_lines, (x1, y1), (x2, y2), color, 1)



    # cv2.imshow("lines1.png",basic_lines)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()


    lines = cv2.HoughLinesP(white_mask, rho=1, theta=np.pi / 180, threshold=1,
                            minLineLength=minHoughLine, maxLineGap=maxGap) #lower angle resolution, less issues with jagged lines? ( they count as one line)
    if lines is None:
        if display:
            cv2.imshow("white mask",white_mask)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        print("NO LINES FOUND")
        return None,None



    
    # bundler = HoughBundler(min_distance=10,min_angle=10)
    print("raw lines")
    print(lines)
    print(f"LINE COUNT:{len(lines)}")
    bundler = HoughBundler(min_distance=200,min_angle=10)
    lines = bundler.process_lines(lines)
    print("bundled lines")
    print(lines)
    print(f"LINE COUNT:{len(lines)}")
    lines = [tuple(line[0]) for line in lines]
    




    filtered_lines = []
    for line in lines:
        x1, y1, x2, y2 = line
        if (math.dist((x1,y1),(x2,y2))>MIN_LINE_LENGTH):
            filtered_lines.append(line)
    
    lines = filtered_lines


    if display:
        basic_lines = np.zeros_like(original)
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line
                # print(line)
                # Generate a random RGB color
                r = random.randint(0, 255)
                g = random.randint(0, 255)
                b = random.randint(0, 255)
                # Convert RGB to BGR for OpenCV
                color = (b, g, r)
                cv2.line(basic_lines, (x1, y1), (x2, y2), color, 1)
                cv2.line(original, (x1, y1), (x2, y2), color, 3)



    print("lines before connecting")
    print(lines)
    print(f"LINE COUNT:{len(lines)}")


    #early exit here!
    if len(lines)>MAX_LINES_TO_PROCESS*1.5:
        return None,None
    lines = get_connected_lines2(lines)
    print(f"FINAL LINE COUNT:{len(lines)}")
    # lines = [line[0] for line in lines]

   
    if display:
        # Create blank image to draw lines
        line_img = np.zeros_like(white_mask)
        color_lines = np.zeros_like(original)
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line
                # print(line)
                # Generate a random RGB color
                r = random.randint(0, 255)
                g = random.randint(0, 255)
                b = random.randint(0, 255)
                # Convert RGB to BGR for OpenCV
                color = (b, g, r)
                cv2.line(line_img, (x1, y1), (x2, y2), 255, 1)
                cv2.line(color_lines, (x1, y1), (x2, y2), color, 1)


    start = time.time()

    best_quad,max_area = get_largest_quad(lines)
    end = time.time()
    print(f"{end - start} seconds")
    # Draw points
    if not best_quad:
        print("TABLE NOT FOUND")
    elif display:
        vert = list(best_quad.exterior.coords)
        vertices = [ (int(round(x)), int(round(y))) for x, y in vert ]
        for point in vertices:
            cv2.circle(original, point, radius=5, color=(0, 255, 0), thickness=-1)  # Green filled circle

    if display:
        cv2.imshow("white mask",white_mask)
        # cv2.imshow("colored lines",color_lines)
        cv2.imshow("Lines before connecting",basic_lines)
        # cv2.imwrite("lines1.png",basic_lines)
        # cv2.imwrite("lines2.png",color_lines)
        
        cv2.imshow("Largest Contour from Lines", original)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    if not best_quad:
        return None,None
    else:
        return best_quad,max_area