import itertools
import math
import random
import time

import cv2
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import shapely

#LINE_CONNECTION_DIST = 100 #max distance between the endpoints of 2 lines which should be connected. this greatly influences the number of potential lines
#MIN_LINE_LENGTH = 50 #the minimum length of a line segment for it to be considered a potential side of the table (before extension)
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
                if self.get_distance(line_2, line_1) < min_distance_to_merge or get_intersection((line_1[0],line_1[1]),(line_1[2],line_1[3]),(line_2[0],line_2[1]),(line_2[2],line_2[3])):
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
    max_connection_dist = max(math.dist(p1,p2), math.dist(p3,p4))
    max_connection_dist /= 10

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
    if (intersection_point and min_dist<max_connection_dist):
        k = pairs.index(min_pair)
        if (k==0 or k==1) and math.dist(p1,intersection_point) < (max_connection_dist*2): #and (math.dist(intersection_point,p2)>math.dist(p1,p2)):
            # new_p1 = intersection_point
            return (1,intersection_point)
        elif (k==2 or k==3) and math.dist(p2,intersection_point) < (max_connection_dist*2): #and (math.dist(p1,intersection_point)>math.dist(p1,p2)):
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
    for combo in itertools.permutations(lines, 3): 
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
    print(line_list)
    if len(line_list)>MAX_LINES_TO_PROCESS:
        return None,None
    
    lines =  [ [(x1, y1), (x2, y2)] for x1, y1, x2, y2 in line_list ]
    line_strings = [shapely.LineString(line) for line in lines]

    max_area = 0
    best_quad = None
    for combo in itertools.combinations(line_strings, 4):  
        # Use polygonize to extract all closed polygons formed by lines
        polygons = shapely.polygonize(combo)
        if not polygons:
            continue
        
        for geom in polygons.geoms:
            poly = shapely.Polygon(geom)
            area = poly.area
            if (len(geom.exterior.coords) - 1 == 4) and shapely.is_simple(geom) and area > max_area:
                max_area = area
                best_quad = poly
    

    if best_quad:
        pass
        print("Max Area:", max_area)
    else:
        return None,None

    return best_quad,max_area

def get_largest_quad3(line_list):
    G = nx.Graph()

    # Build graph: endpoints as nodes, segments as edges
    for x1, y1, x2, y2 in line_list:
        p1 = (x1, y1)
        p2 = (x2, y2)
        G.add_edge(p1, p2)

    seen = set()
    max_area = 0
    best_quad = None

    # Use cycle_basis to find all simple cycles
    for cycle in nx.cycle_basis(G):
        if len(cycle) != 4:
            continue
        
        print(cycle)
        # Canonical form to avoid duplicates (e.g., rotations, flips)
        canon = tuple(sorted(cycle))
        print(f"canon:{canon}")
        if canon in seen:
            continue
        seen.add(canon)
        poly = shapely.Polygon(cycle)
        if not poly.is_valid or not poly.is_simple:
            continue

        area = poly.area
        if area > max_area:
            max_area = area
            best_quad = poly

    if best_quad:
        print("Max Area:", max_area)
        return best_quad, max_area
    else:
        return None, None


def get_largest_quad2(line_list):
    G = nx.Graph()

    # Add edges to graph from line segments
    for x1, y1, x2, y2 in line_list:
        p1 = (x1, y1)
        p2 = (x2, y2)
        G.add_edge(p1, p2)

    max_area = 0
    best_quad = None

    for cycle in nx.simple_cycles(G,length_bound=4):
        if len(cycle) != 4:
            continue

        # Sort the cycle to avoid duplicates (rotate and flip to canonical form)
        # canon = tuple(sorted(cycle))

        # print(cycle)
        # Create polygon and validate
        poly = shapely.Polygon(cycle)
        
        if not poly.is_valid or not poly.is_simple:
            continue

        area = poly.area
        if area > max_area:
                max_area = area
                best_quad = poly
    

    if best_quad:
        print("Max Area:", max_area)
        return best_quad, max_area
    else:
        return None, None



# def get_largest_quad(line_list):
#     if len(line_list) > MAX_LINES_TO_PROCESS:
#         return None, None

#     # Convert raw line data directly to LineStrings
#     line_strings = [shapely.LineString([(x1, y1), (x2, y2)]) for x1, y1, x2, y2 in line_list]

#     # Use polygonize once on the full list
#     polygons = list(shapely.polygonize(line_strings))

#     max_area = 0
#     best_quad = None

#     for poly in polygons:
#         if not isinstance(poly, shapely.Polygon):
#             continue

#         coords = list(poly.exterior.coords)
        
#         # Check for quadrilateral: 4 edges = 5 coords (last repeats first)
#         if len(coords) - 1 == 4 and poly.is_valid and poly.is_simple:
#             area = poly.area
#             if area > max_area:
#                 max_area = area
#                 best_quad = poly

#     if best_quad:
#         print("Max Area:", max_area)
#         return best_quad, max_area
#     else:
#         return None, None


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


def get_top_longest_lines(segments):
    # Compute length for each line segment and pair it with the segment
    segments_with_length = [
        (math.hypot(x2 - x1, y2 - y1), [x1, y1, x2, y2])
        for x1, y1, x2, y2 in segments
    ]
    
    # Sort by length in descending order
    segments_with_length.sort(reverse=True, key=lambda x: x[0])
    
    # Calculate number of lines to take: top 30% or at least 30
    n = min(30, len(segments) * 30 // 100)
    
    # Get the top segments and return only the line definitions (not lengths)
    top_segments = [seg for _, seg in segments_with_length[:n]]
    
    return top_segments


def find_white_lines_and_largest_contour(white_mask,original,maxGap=3,minHoughLine=15,display=False):
    asd = original.copy()
    scaling_ratio = original.shape[0] / 1080
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


    lines = cv2.HoughLinesP(white_mask, rho=1, theta=np.pi / 180, threshold=int(50*scaling_ratio),
                            minLineLength=minHoughLine, maxLineGap=maxGap) #lower angle resolution, less issues with jagged lines? ( they count as one line)
    if lines is None:
        # if display:
        #     cv2.imshow("white mask",white_mask)
        #     cv2.waitKey(0)
        #     cv2.destroyAllWindows()
        print("NO LINES FOUND")
        return None,None



    
    # bundler = HoughBundler(min_distance=10,min_angle=10)

    print("raw lines")
    # print(lines)
    print(f"LINE COUNT:{len(lines)}")


    #this is so stupid LOL
    # if len(lines)>250:
    #     return None,None
    


    bundler = HoughBundler(min_distance=100*scaling_ratio,min_angle=5)

    start = time.time()
    lines = bundler.process_lines(lines)
    end = time.time()
    print(f"bundling:{end - start} seconds")

    # print("bundled lines")
    # print(lines)
    # print(f"LINE COUNT:{len(lines)}")

    lines = [tuple(line[0]) for line in lines]
    




    # filtered_lines = []
    # for line in lines:
    #     x1, y1, x2, y2 = line
    #     if (math.dist((x1,y1),(x2,y2))>MIN_LINE_LENGTH):
    #         filtered_lines.append(line)
    
    # lines = filtered_lines



    # print(f"nr of lines before picking top:{len(lines)}")
    # final_lines = get_top_longest_lines(lines)
    # lines = final_lines


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
                # cv2.line(basic_lines, (x1, y1), (x2, y2), color, 3)
                cv2.line(original, (x1, y1), (x2, y2), color, 3)



    # print("lines before connecting")
    # print(lines)
    # print(f"LINE COUNT:{len(lines)}")

    # print(f"nr of lines before connecting:{len(lines)}")
    
    print(original.shape[0])
    print(f"before filtering:{len(lines)}")
    filtered_lines = []
    for line in lines:
        x1, y1, x2, y2 = line
        if (math.dist((x1,y1),(x2,y2))> original.shape[0]/15 ):
            filtered_lines.append(line)
    
    lines = filtered_lines
    print(f"after filtering:{len(lines)}")




    print(f"nr of lines before extending:{len(lines)}")
    start = time.time()
    lines = get_connected_lines2(lines)
    end = time.time()
    print(f"extending:{end - start} seconds")


    
    
    
    


    if display:
        # Create blank image to draw lines
        line_img = np.zeros_like(white_mask)
        color_lines = np.zeros_like(asd)
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

    best_quad,max_area = get_largest_quad3(lines)
    end = time.time()
    print(f"finding quad:{end - start} seconds")
    # Draw points
    if not best_quad:
        pass
        # print("TABLE NOT FOUND")
    elif display:
        vert = list(best_quad.exterior.coords)
        vertices = [ (int(round(x)), int(round(y))) for x, y in vert ]
        print(vertices)
        for point in vertices:
            cv2.circle(original, point, radius=5, color=(0, 255, 0), thickness=-1)  # Green filled circle

    if display:
        print(f"display:{display}")
        cv2.imshow("white mask",white_mask)
        cv2.imshow("colored lines(after connecting)",color_lines)
        # cv2.imshow("Lines before connecting",basic_lines)
        # cv2.imwrite("lines1.png",basic_lines)
        # cv2.imwrite("lines2.png",color_lines)
        
        cv2.imshow("Lines before connecting??", original)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    if not best_quad:
        return None,None
    else:
        return best_quad,max_area
    
    
    
    
    
def find_table(img, display=False):
#am I calculating edges from the right image?
#1. picking saturation threshold so I get all the table lines without too much random shit (automatic or connect to stronger parts?)
#2. how to remove double lines?
#3. speed up quad finding (only look at the longest lines, quick test?)
#4. add conditions to check if it's probably the table?
    scaling_ratio = img.shape[0] / 1080
    print(f"SCALING RATIO:{scaling_ratio}")
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    v_channel = hsv[:, :, 2]
    s_channel = hsv[:, :, 1]

    v_thresh = np.percentile(v_channel, 50)
    s_thresh = np.percentile(s_channel, 50)

    edges = cv2.Canny(img,80,200)

    only_edge_parts = cv2.bitwise_and(img, img, mask=edges)
    hsv_edges =  cv2.cvtColor(only_edge_parts, cv2.COLOR_BGR2HSV)

    ve_channel = hsv_edges[:, :, 2]
    se_channel = hsv_edges[:, :, 1]

    print('THRESHOLDS')
   

    # Flatten the V channel and filter out zeros
    ve_values = ve_channel.flatten()
    ve_nonzero = ve_values[ve_values > 0]

    se_values = se_channel.flatten()
    se_nonzero = se_values[se_values > 0]

    ve_thresh = np.percentile(ve_nonzero, 50)
    se_thresh = np.percentile(se_nonzero, 50)

    print(ve_thresh)
    print(se_thresh)

    # Create binary map: bright AND low saturation

    # binary_map = (v_channel >= v_thresh) & (s_channel < 170)
    binary_map = (v_channel >= ve_thresh) & (s_channel <= se_thresh)
    # Convert boolean mask to uint8 (0 or 1)
    binary_map_uint8 = binary_map.astype(np.uint8)
    binary_image = (binary_map.astype(np.uint8)) * 255



    white = cv2.bitwise_and(img, img, mask=binary_image)



    mask = cv2.blur(binary_image,(3,3))
    # mask = binary_image


    thin_edges = cv2.ximgproc.thinning(edges,thinningType=0)
    thin_mask = cv2.ximgproc.thinning(mask,thinningType=0)
    # combined2 = cv2.bitwise_and(edges,mask)

    #TESTING!!!
    combined2 = edges

    if display:
        cv2.imshow('original',img)
        cv2.imshow('mask',mask)
        cv2.imshow('edge',edges)

        # # Show the result
        cv2.imshow('Top 20% Brightest Pixels (V channel)', combined2)

        cv2.waitKey(0)
        cv2.destroyAllWindows()


    best_quad = None
    max_area = 0

    largest_area = 0
    largest_vert = []
    largest_quad = None

    start = time.time()

    minHough = [60,50,40,30]
    gaps = [4,5,6,7,10]

    minHough = [int(x*scaling_ratio) for x in minHough]
    gaps = [int(x*scaling_ratio) for x in gaps]


    #threshold and angle resolution parameters?

    original = img


    for combo in itertools.product(gaps,minHough):
        gap = combo[0]
        minH = combo[1]
        
        img_copy = img.copy()
        quad,area = find_white_lines_and_largest_contour(combined2,img_copy,maxGap=gap,minHoughLine=minH,display=False)
        
        if quad and area>largest_area:
            largest_quad = quad
            largest_vert = list(quad.exterior.coords)
            largest_area = area
            best_combo = combo


    end = time.time()
    print(f"{end - start} seconds")

    if largest_area>0:
    
        vertices = [ (int(round(x)), int(round(y))) for x, y in largest_vert ]
        print(vertices)

        if display:
            cv2.namedWindow("table", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("table", 800, 600)
            for point in vertices:
                cv2.circle(original, point, radius=5, color=(0, 255, 0), thickness=-1)  # Green filled circle

            cv2.imshow("table",original)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        return largest_quad

    else:
        print("TABLE NOT FOUND")
        return None