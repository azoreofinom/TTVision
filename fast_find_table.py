import itertools
import math
import random
import time

import cv2
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import shapely
import cProfile
import pstats





#Used to determine if 2 line segments belong to the same line
MAX_BUNDLING_ANGLE = 5 
MAX_BUNDLING_DISTANCE = 50
MAX_LAB_COLOR_DIST = 50



COLLINEARITY_DIST = 5

#Percentage based distance limit for connecting 2 lines
LINE_EXTENSION_PERCENTAGE = 0.5

#Absolute distance limit for connecting 2 lines
LINE_EXTENSION_LIMIT = 50

MIN_ORIENTATION_DIFF = 1

class Line:
    def __init__(self, ends, color, orientation):
        self.ends = ends
        self.color = color
        self.orientation = orientation

class HoughBundler:     
    def __init__(self,lab_img, max_distance=5, max_angle=2):
        self.max_distance = max_distance
        self.max_angle = max_angle
        self.lab_img = lab_img

    def get_orientation(self,line):
        orientation = math.atan2((line[3] - line[1]), (line[2] - line[0]))
        # print(f"raw:{math.degrees(orientation)}")
        return (math.degrees(orientation) % 180)
    
    
    def are_collinear(self,l1,l2):
        x0,y0,_,_ = l2
        x1, y1, x2, y2 = l1

        lmag = math.sqrt(math.pow((x2 - x1), 2) + math.pow((y2 - y1), 2))
        perpendicular_distance = abs((y2-y1)*x0 - (x2-x1)*y0 + x2*y1 - y2*x1)/lmag
        return perpendicular_distance <= COLLINEARITY_DIST
    
    

    
    def check_is_line_different(self, line_1, groups, max_distance_to_merge, max_angle_to_merge):
        orientation_1 = line_1.orientation
        not_found_group = True
        avg_line_color = line_1.color
       
        for group in groups:
            orientation_2 = group[0].orientation
           
            group_color = group[-1].color

            large_angle = max(orientation_1,orientation_2)
            small_angle = min(orientation_1,orientation_2)
            orientation_diff = min(abs(orientation_1 - orientation_2), (180-large_angle)+small_angle)
          

            if  orientation_diff > max_angle_to_merge or not self.are_collinear(group[0].ends,line_1.ends)  or  abs(math.dist(avg_line_color,group_color )) > MAX_LAB_COLOR_DIST:
                continue
            for line_2 in group:
        
                if self.get_distance(line_2.ends, line_1.ends) < max_distance_to_merge or get_intersection((line_1.ends[0],line_1.ends[1]),(line_1.ends[2],line_1.ends[3]),(line_2.ends[0],line_2.ends[1]),(line_2.ends[2],line_2.ends[3])):
                    group.append(line_1)
                    not_found_group = False
                    break
                    # return False
                            
        return not_found_group


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

        # nr_of_bins = int(180/self.max_angle)
        # bins = [ [] for _ in range(nr_of_bins) ]
        
        for line_new in lines[1:]:
            if self.check_is_line_different(line_new, groups, self.max_distance, self.max_angle):
                groups.append([line_new])

        return groups

    def merge_line_segments(self, lines):
        orientation = lines[0].orientation
      
        if(len(lines) == 1):
            return np.block([[lines[0].ends[:2], lines[0].ends[2:]]])

        points = []
        for line_object in lines:
            line = line_object.ends
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
        # print(lines[0])
        # line_objects = []
        # for line in lines:
        #     line_objects.append(Line(line,average_color_on_line(self.lab_img,line)))

        lines_horizontal  = []
        lines_vertical  = []
  
        for line_i in [l[0] for l in lines]:
            # print(line_i)
            orientation = self.get_orientation(line_i)
            # print(orientation)
            # if vertical
            if 45 < orientation <= 90:
                lines_vertical.append(Line(line_i,average_color_on_line(self.lab_img,line_i),orientation))
            else:
                lines_horizontal.append(Line(line_i,average_color_on_line(self.lab_img,line_i),orientation))

        lines_vertical  = sorted(lines_vertical , key=lambda line: line.ends[1])
        lines_horizontal  = sorted(lines_horizontal , key=lambda line: line.ends[0])
        merged_lines_all = []

        # for each cluster in vertical and horizantal lines leave only one line
        for i in [lines_horizontal, lines_vertical]:
            if len(i) > 0:
                groups = self.merge_lines_into_groups(i)
                merged_lines = []
                for group in groups:
                    # if len(group)>1:
                    #     print(group)
                    merged_lines.append(self.merge_line_segments(group))
                merged_lines_all.extend(merged_lines)
                    
        return np.asarray(merged_lines_all)
    




def get_intersection(p1, p2, p3, p4):
    """Returns the intersection point of line segments p1–p2 and p3–p4 as (x, y), or None if no intersection."""
    x1, y1 = map(float, p1)
    x2, y2 = map(float, p2)
    x3, y3 = map(float, p3)
    x4, y4 = map(float, p4)

    denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if abs(denom) < 1e-12:
        return None  # Parallel or coincident lines

    # Compute intersection point using determinant form
    det1 = x1 * y2 - y1 * x2
    det2 = x3 * y4 - y3 * x4

    px = (det1 * (x3 - x4) - (x1 - x2) * det2) / denom
    py = (det1 * (y3 - y4) - (y1 - y2) * det2) / denom

    # Check if intersection point lies within both line segments
    if (
        min(x1, x2) - 1e-9 <= px <= max(x1, x2) + 1e-9 and
        min(y1, y2) - 1e-9 <= py <= max(y1, y2) + 1e-9 and
        min(x3, x4) - 1e-9 <= px <= max(x3, x4) + 1e-9 and
        min(y3, y4) - 1e-9 <= py <= max(y3, y4) + 1e-9
    ):
        return int(px), int(py)
    return None



def get_line_intersection(p1, p2, p3, p4):
    """
    Returns the intersection point (px, py) of infinite lines p1–p2 and p3–p4.
    Returns None if the lines are parallel or coincident.
    """
    # Convert to floats to avoid integer overflow
    x1, y1 = map(float, p1)
    x2, y2 = map(float, p2)
    x3, y3 = map(float, p3)
    x4, y4 = map(float, p4)

    denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if abs(denom) < 1e-12:
        return None, None  # Parallel or coincident lines

    det1 = x1 * y2 - y1 * x2
    det2 = x3 * y4 - y3 * x4

    px = (det1 * (x3 - x4) - (x1 - x2) * det2) / denom
    py = (det1 * (y3 - y4) - (y1 - y2) * det2) / denom

    inside_line = False
    if (
        min(x1, x2) - 1e-9 <= px <= max(x1, x2) + 1e-9 and
        min(y1, y2) - 1e-9 <= py <= max(y1, y2) + 1e-9 and
        min(x3, x4) - 1e-9 <= px <= max(x3, x4) + 1e-9 and
        min(y3, y4) - 1e-9 <= py <= max(y3, y4) + 1e-9
    ):
        inside_line = True

    return inside_line, (int(px), int(py))


         

def dist_between_lines(p1,p2,p3,p4):
    pairs = [(p1,p3),(p1,p4),(p2,p3),(p2,p4)]
    # Find the smallest distance and the corresponding pair
    min_dist = float('inf')

    for a, b in pairs:
        dist = math.dist(a, b)
        if dist < min_dist:
            min_dist = dist
    
    return min_dist

def get_line_color(line, lab_img):
    (x1, y1, x2, y2) = line
    color1 = lab_img[y1, x1].astype(np.float32)
    color2 = lab_img[y2, x2].astype(np.float32)
    result = ((color1 + color2) / 2).tolist()
    return result


def average_color_on_line(img, line):
    (x1, y1, x2, y2) = line

    # Get points along the line using Bresenham's algorithm
    line_points = np.linspace((x1, y1), (x2, y2), num=int(np.hypot(x2-x1, y2-y1)))
    line_points = np.round(line_points).astype(int)

    # Clip to image bounds
    h, w = img.shape[:2]
    line_points = line_points[
        (line_points[:, 0] >= 0) & (line_points[:, 0] < w) &
        (line_points[:, 1] >= 0) & (line_points[:, 1] < h)
    ]

    colors = img[line_points[:, 1], line_points[:, 0]]
    avg_color = colors.mean(axis=0)

    return tuple(map(int, avg_color[::-1]))  # Convert BGR→RGB if needed



def get_orientation(line):
        orientation = math.atan2((line[3] - line[1]), (line[2] - line[0]))
        # print(f"raw:{math.degrees(orientation)}")
        return (math.degrees(orientation) % 180)


def get_connected_lines(lines,lab_img):
    final_lines = []

    line_objects = []
    for line in lines:
        orientation = get_orientation(line)
        color = average_color_on_line(lab_img,line)
        line_objects.append(Line(line,color,orientation))

    #only considering nearby lines for each line

    for line_object in line_objects:
        (x1, y1, x2, y2)  = line_object.ends
        p1, p2 = (x1, y1), (x2, y2)
        neighbours = []
        line_color = line_object.color
        max_connection_dist =  min(math.dist(p1,p2) * LINE_EXTENSION_PERCENTAGE,LINE_EXTENSION_LIMIT)
        line_orientation = line_object.orientation
        for nei_object in line_objects:
            if line_object != nei_object:
                (x1, y1, x2, y2)  = nei_object.ends
                p3, p4 = (x1, y1), (x2, y2)
                dist = dist_between_lines(p1,p2,p3,p4)
                nei_color = nei_object.color
                nei_orientation = nei_object.orientation

                large_angle = max(line_orientation,nei_orientation)
                small_angle = min(line_orientation,nei_orientation)
                orientation_diff = min(abs(line_orientation - nei_orientation), (180-large_angle)+small_angle)
                
                if dist < max_connection_dist and orientation_diff>MIN_ORIENTATION_DIFF:  #and abs(math.dist(line_color,nei_color )) < MAX_LAB_COLOR_DIST:
                # if (dist < max_connection_dist or get_intersection(p1,p2,p3,p4) is not None) and orientation_diff>MIN_ORIENTATION_DIFF:        #and math.dist(line_color,nei_color)<MAX_LAB_COLOR_DIST:
                    neighbours.append(nei_object.ends)

        
        for combo in itertools.combinations(neighbours, 2):
            p3, p4 = (combo[0][0],combo[0][1]), (combo[0][2],combo[0][3])
            p5, p6 = (combo[1][0],combo[1][1]), (combo[1][2],combo[1][3])
            inter1 = get_extension_side2(p1,p2,p3,p4,max_connection_dist)
            inter2 = get_extension_side2(p1,p2,p5,p6,max_connection_dist)
     

            if inter1 and inter2 and inter1[0] != inter2[0]:
            
                final_lines.append((*inter1[1],*inter2[1]))
               

                # inter1_color = lab_img[inter1[1][1]][inter1[1][0]]
                # inter2_color = lab_img[inter2[1][1]][inter2[1][0]]
                # # print(inter1_color)
                # if math.dist(line_color,inter1_color)<MAX_LAB_COLOR_DIST and math.dist(line_color,inter2_color)<MAX_LAB_COLOR_DIST:
                #     final_lines.append((*inter1[1],*inter2[1]))

      
    return final_lines



def get_extension_side(p1,p2,p3,p4, max_connection_dist):
    # max_connection_dist = max(math.dist(p1,p2), math.dist(p3,p4)) / 10

    pairs = [(p1,p3),(p1,p4),(p2,p3),(p2,p4)]
    # Find the smallest distance and the corresponding pair
    min_dist = float('inf')
    min_pair = None

    for a, b in pairs:
        dist = math.dist(a, b)
        if dist < min_dist:
            min_dist = dist
            min_pair = (a, b)
    
    inside_line, intersection_point = get_line_intersection(p1,p2,p3,p4)
    if (intersection_point and min_dist<max_connection_dist):
        k = pairs.index(min_pair)
        if (k==0 or k==1) and math.dist(p1,intersection_point) < (max_connection_dist*2): #and (math.dist(intersection_point,p2)>math.dist(p1,p2)):
            return (1,intersection_point)
        elif (k==2 or k==3) and math.dist(p2,intersection_point) < (max_connection_dist*2): #and (math.dist(p1,intersection_point)>math.dist(p1,p2)):
            return (2,intersection_point)
        else:
            return None
    else:
        return None
    

def get_extension_side2(p1,p2,p3,p4, max_connection_dist): 
    inside_line, intersection_point = get_line_intersection(p1,p2,p3,p4)
    if (intersection_point):
        dist1 = math.dist(p1,intersection_point)
        dist2 = math.dist(p2,intersection_point)

        if dist1 <= dist2:
            side = 1
            min_dist = dist1
        else:
            side = 2
            min_dist = dist2

        if inside_line or min_dist < (max_connection_dist*2):
            if side==1:
                return (1,intersection_point)
            else:
                return (2,intersection_point)
        else:
            return None

  
    else:
        return None


def get_largest_quad(line_list):
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




def find_white_lines_and_largest_contour(white_mask,original,maxGap=3,minHoughLine=15,display=False):
    asd = original.copy()
    scaling_ratio = original.shape[0] / 1080
    

    all_lines = []
    thetas = [np.pi / 480, np.pi / 180]
    ths = [100*scaling_ratio, 50*scaling_ratio]

    for angle in thetas:
        for h in ths:
            lines = cv2.HoughLinesP(white_mask, rho=1, theta=angle, threshold=int(h*scaling_ratio),
                                    minLineLength=minHoughLine, maxLineGap=maxGap)
            all_lines.extend(lines)

    lines = all_lines

    if lines is None:
        print("NO LINES FOUND")
        return None,None

    
    print(f"raw LINE COUNT:{len(lines)}")


    if display:
        # Create blank image to draw lines
        line_img = np.zeros_like(white_mask)
        raw_color_lines = np.zeros_like(asd)
        if lines is not None:
            for line in lines:
               
                x1, y1, x2, y2 = line[0]
                # Generate a random RGB color
                r = random.randint(0, 255)
                g = random.randint(0, 255)
                b = random.randint(0, 255)
                # Convert RGB to BGR for OpenCV
                color = (b, g, r)
                # cv2.line(line_img, (x1, y1), (x2, y2), 255, 1)
                cv2.line(raw_color_lines, (x1, y1), (x2, y2), color, 1)


    lab_img = cv2.cvtColor(original, cv2.COLOR_BGR2LAB)
    # lab_img = cv2.GaussianBlur(lab_img, (3, 3), 1.5)

    bundler = HoughBundler(lab_img, max_distance=MAX_BUNDLING_DISTANCE*scaling_ratio,max_angle=MAX_BUNDLING_ANGLE)

    # old_len = 0
    # new_len = len(lines)

    # while new_len != old_len:
    #     old_len = new_len
    #     lines = bundler.process_lines(lines)
    #     new_len = len(lines)

    lines = bundler.process_lines(lines)
    lines = bundler.process_lines(lines)

    # lines = bundler.process_lines(lines)

   
    print(f"bundled LINE COUNT:{len(lines)}")

  
    lines = [tuple(line[0]) for line in lines]
    


    print(f"before filtering:{len(lines)}")
    filtered_lines = []
    for line in lines:
        x1, y1, x2, y2 = line
        if (math.dist((x1,y1),(x2,y2))> original.shape[0]/15 ):
            filtered_lines.append(line)
    
    lines = filtered_lines
    print(f"after filtering:{len(lines)}")



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
                    cv2.line(original, (x1, y1), (x2, y2), color, 2)



    print(f"nr of lines before extending:{len(lines)}")
    lines = get_connected_lines(lines,lab_img)
    
    print(f"nr of lines after extending:{len(lines)}")


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


  
    best_quad,max_area = get_largest_quad(lines)

    # Draw points
    if not best_quad:
        print("TABLE NOT FOUND")
    elif display:
        vert = list(best_quad.exterior.coords)
        vertices = [ (int(round(x)), int(round(y))) for x, y in vert ]
        # print(vertices)
        for point in vertices:
            cv2.circle(original, point, radius=5, color=(0, 255, 0), thickness=-1)  # Green filled circle

    if display:
        cv2.imshow("lines before bundling",raw_color_lines)
        cv2.imshow("colored lines(after connecting)",color_lines)
        cv2.imshow("Lines after bundling", original)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    if not best_quad:
        return None,None
    else:
        return best_quad,max_area
    
    
    
    
    
def find_table(img, display=False,showresult=False):

    scaling_ratio = img.shape[0] / 1080
    image_area = img.shape[0] * img.shape[1]
    print(f"SCALING RATIO:{scaling_ratio}")
    
    edges = cv2.Canny(img,150,200)
    # cv2.imshow("edges",edges)
    kernel = np.ones((3,3),np.uint8)
    # kernel = np.ones((2,2),np.uint8)
    edges = cv2.dilate(edges,kernel,iterations = 1)
    # cv2.imshow("dilation",edges)

    combined2 = edges


    largest_area = 0
    largest_vert = []
    largest_quad = None



    minHough = [50]
    gaps = [5]

    minHough = [int(x*scaling_ratio) for x in minHough]
    gaps = [int(x*scaling_ratio) for x in gaps]



    original = img


    for combo in itertools.product(gaps,minHough):
        print(combo)
        gap = combo[0]
        minH = combo[1]
        
        img_copy = img.copy()
        quad,area = find_white_lines_and_largest_contour(combined2,img_copy,maxGap=gap,minHoughLine=minH,display=display)
        
        if quad and area>largest_area:
            largest_quad = quad
            largest_vert = list(quad.exterior.coords)
            largest_area = area


    if largest_area>0 and largest_area/image_area > 0.02:
    
        vertices = [ (int(round(x)), int(round(y))) for x, y in largest_vert ]
        #print(vertices)

        if showresult:
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


if __name__ == '__main__':
    
    #balanced.png doesn't work, table lines up with another line ok now
    #blue.png.., hall.jpg


    profiler = cProfile.Profile()
    # img = cv2.imread("images/hall18.jpg")
    img = cv2.imread("images/tabletest2.png")
    profiler.enable()
    find_table(img,display=True,showresult=True)
    #asd.find_table(img,display=True)
    profiler.disable()
    stats = pstats.Stats(profiler).sort_stats('cumtime')
    stats.print_stats(30)