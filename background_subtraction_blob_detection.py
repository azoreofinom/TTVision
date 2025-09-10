import cv2
import numpy as np
import math
import collections
import time
import find_table
import heapq
import shapely
import stats

class BallCandidate:
    def __init__(self, radius: float, circularity: float, position: tuple):
        self.radius = radius
        self.circularity = circularity
        self.position = position  # position should be a tuple (x, y)

    def __repr__(self):
        return (f"BallCandidate(radius={self.radius}, "
                f"circularity={self.circularity}, "
                f"position={self.position})")

# backSub = cv2.bgsegm.createBackgroundSubtractorMOG(history=20,nmixtures=5, backgroundRatio=0.7) #thrash
backSub = cv2.createBackgroundSubtractorMOG2(varThreshold=16, detectShadows=True, history=500) #best so far
# backSub = cv2.createBackgroundSubtractorKNN(history=500, dist2Threshold=100, detectShadows=True) #ok
# backSub = cv2.bgsegm.createBackgroundSubtractorGMG(initializationFrames=120, decisionThreshold=0.9)
# backSub = cv2.bgsegm.createBackgroundSubtractorGSOC() #sees the ball in the problematic serve case... could be good with some tuning?



DOWNSAMPLE_ROWS = 360
DOWNSAMPLE_COLS = 640

# DOWNSAMPLE_ROWS = 1080
# DOWNSAMPLE_COLS = 1920


MIN_DIST_FROM_OTHER_CONTOURS = DOWNSAMPLE_COLS //15 

OVERLAPPING_CONTOUR_DIST = 10 /2
MIN_RADIUS = 1.5 # this is a bit scuffed...
MAX_RADIUS = 30 / 2
MIN_CIRCULARITY = 0.4

MAX_DIST_FROM_PREVIOUS_POS  = DOWNSAMPLE_COLS // 20




TRACKING_WINDOW_HEIGHT = DOWNSAMPLE_ROWS // 4
TRACKING_WINDOW_WIDTH = DOWNSAMPLE_COLS // 4





# MIN_DIST_FROM_OTHER_CONTOURS = 30 
# OVERLAPPING_CONTOUR_DIST = 10 
# MIN_RADIUS = 4 
# MAX_RADIUS = 30 
# MIN_CIRCULARITY = 0.65 

def vector_length(v):
    return math.sqrt(sum(coord ** 2 for coord in v))

def get_ball_contour(frame_mask):
    #serves close to the body might be a problem
    #is there even a point to the near-criteria?
    #the ball might go out of frame during serves!
    #we ocasionally lose the ball... account for that
    #shadow of the ball on the table LOL
    #circ > 0.7
    #decrease dist threhsold
    #after contact with bat , ball won't be visible for 3/4 frames
    #if two contours with similar, high circularity, focus on bigger one, or the one consistent with expected motion?
    #r>5 and circularity>0.4?
    #make sure chosen contour is not near another large contour
    #pixels should be mostly bright
    
    #eliminate all contours which don't fit area criteria first,then do centroid dist, and only then do point based
    # img_int32 = np.int32(frame_mask)


    contours, _ = cv2.findContours(frame_mask, cv2.RETR_EXTERNAL     , cv2.CHAIN_APPROX_SIMPLE)
    contour_image = np.zeros_like(frame_mask)
    cv2.drawContours(contour_image, contours, -1, (255), 1)  # 1 is line thickness




    # if contours:
    #     largest_contour = max(contours, key=cv2.contourArea)
    #     cv2.drawContours(contour_image, [largest_contour], -1, (255), 25)
    # else:
    #     print("No contours found.")


    # cv2.imshow('big',contour_image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    best_contour = None
    best_score = 0  # Score based on circularity

    not_ball_cnt = set()
    for i,cnt in enumerate(contours):
        area = cv2.contourArea(cnt)
        radius = math.sqrt(area / math.pi)
        perimeter = cv2.arcLength(cnt, True)
        if perimeter==0:
            not_ball_cnt.add(i)
            continue

        circularity = 4 * math.pi * area / (perimeter ** 2)
        # perimeter = cv2.arcLength(cnt, True)
        if area == 0 or radius >= MAX_RADIUS or radius<MIN_RADIUS or circularity<0.85:
            not_ball_cnt.add(i)
      
    

    #was checking to see if the contour was too close to other contours. ignore for now...
    # for i,cnt1 in enumerate(contours):
    #     for j,cnt2 in enumerate(contours):
    #         if i!=j and i not in not_ball_cnt:
    #             # print(f"cnt1:{cnt1}")
    #             centroid1 = get_cnt_centroid(cnt1)
    #             # print(f"centroid:{centroid1}")
    #             # print(f"cnt2:{cnt2}")
    #             centroid2 = get_cnt_centroid(cnt2)
    #             # print(f"centroid:{centroid2}")
    #             if math.dist(centroid1,centroid2) > OVERLAPPING_CONTOUR_DIST and math.dist(centroid1,centroid2) < MIN_DIST_FROM_OTHER_CONTOURS:
    #                 not_ball_cnt.add(i)
    #                 not_ball_cnt.add(j)

   
    # print(f"nr of candidates:{len(contours) - len(not_ball_cnt)}")

    for i,cnt in enumerate(contours):
        if i in not_ball_cnt:
            continue

        area = cv2.contourArea(cnt)
        if area == 0:
            continue

        radius = math.sqrt(area / math.pi)
        perimeter = cv2.arcLength(cnt, True)
        if perimeter == 0:
            continue

        circularity = 4 * math.pi * area / (perimeter ** 2)

        if circularity > best_score:
            best_score = circularity
            best_contour = cnt

    return best_contour


def get_ball_during_point(frame_mask,prev_ball,left,right,roi_l,roi_r,ball_area):
    #find contour in roi which is closest in radius and circ to previous? circ closeness is not reliable!
    #roi needs to be large enough to allow for change of direction after hit
    #extension: instead of simple roi, consider parabola?
    prev_r = prev_ball.radius
    prev_c = prev_ball.circularity
    prev_pos = prev_ball.position
    print(prev_pos)

    contours, _ = cv2.findContours(frame_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)


    # roi_topleft = (max(0, prev_pos[0]-TRACKING_WINDOW_WIDTH), max(0,prev_pos[1]-TRACKING_WINDOW_HEIGHT))
    roi_topleft = roi_l
    roi_botright = roi_r



    # roi = frame_mask[roi_topleft[1]:roi_botright[1], roi_topleft[0]:roi_botright[0]]
    # contours, _ = cv2.findContours(roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE, offset=roi_topleft)



    

    # print(roi.shape)
    

    # frame_mask = cv2.cvtColor(frame_mask,cv2.COLOR_GRAY2RGB)
    # cv2.rectangle(frame_mask, roi_topleft,roi_botright, (0,255,0),3)
    # # cv2.imshow("roi",roi)
   
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    best_contour = None
    lowest_diff = float('inf') 

    not_ball_cnt = set()
    # for i,cnt in enumerate(contours):
    #     area = cv2.contourArea(cnt)
    #     radius = math.sqrt(area / math.pi)
    #     perimeter = cv2.arcLength(cnt, True)
    #     if perimeter==0:
    #         not_ball_cnt.add(i)
    #         continue

    #     circularity = 4 * math.pi * area / (perimeter ** 2)
    #     if area == 0 or area < ball_area/2 or area>ball_area*4:
    #         not_ball_cnt.add(i)
    
    for i,cnt1 in enumerate(contours):
        for j,cnt2 in enumerate(contours):
            if i!=j and i not in not_ball_cnt:
                centroid1 = get_cnt_centroid(cnt1)
                centroid2 = get_cnt_centroid(cnt2)
                nearby_area = cv2.contourArea(cnt2)
                if  math.dist(centroid1,centroid2) < MIN_DIST_FROM_OTHER_CONTOURS and nearby_area>ball_area*4:
                    not_ball_cnt.add(i)
                    break
                    # not_ball_cnt.add(j)
      

    for i,cnt in enumerate(contours):
        pos = get_cnt_centroid(cnt)

        if i in not_ball_cnt:
            continue
        

        #only check "above" the table, should reduces FPs by a lot
        if not (pos[0]>=left and pos[0]<=right):
            continue

        if not (pos[0]>=roi_topleft[0] and pos[0]<=roi_botright[0] and pos[1]>=roi_topleft[1] and pos[1]<=roi_botright[1]):
            continue
        
        area = cv2.contourArea(cnt)
        radius = math.sqrt(area / math.pi)
        perimeter = cv2.arcLength(cnt, True)
        if perimeter == 0:
            continue
        
        circularity = 4 * math.pi * area / (perimeter ** 2)

        if area == 0 or area < ball_area/2 or area>ball_area*4 or circularity<0.4: #or radius >= MAX_RADIUS or radius<MIN_RADIUS or circularity<0.45:
            continue

        difference_score = abs(radius - prev_r) + abs(circularity - prev_c) * 10

        if difference_score < lowest_diff:
            lowest_diff = difference_score
            best_contour = cnt

    return best_contour


def get_cnt_centroid(contour):
   if contour is None:
       return (-1,-1)
   M = cv2.moments(contour)
   if M["m00"]==0:  
       return (contour[0][0][0],contour[0][0][1])
   cX = int(M["m10"] / M["m00"])
   cY = int(M["m01"] / M["m00"])
   return cX,cY

def get_cnt_bottom(cnt):
    return tuple(cnt[cnt[:,:,1].argmax()][0])

def get_connected_contours(contours):
    connected_cnt = set()
    for i,cnt1 in enumerate(contours):
        for j,cnt2 in enumerate(contours):
            if i!=j and not (i in connected_cnt and j in connected_cnt) and contours_are_near(cnt1,cnt2):
                connected_cnt.add(i)
                connected_cnt.add(j)

    return connected_cnt


def contours_are_near(cnt1,cnt2):
    for c1 in cnt1:
        for c2 in cnt2:
            p1 = tuple(c1.flatten())
            p2 = tuple(c2.flatten())
            if math.dist(p1,p2) < MIN_DIST_FROM_OTHER_CONTOURS:
                return True
    return False


def is_close_to_others(ball_cnt, cnt_mask):
    rows, cols = cnt_mask.shape[0],cnt_mask.shape[1]
    dirs = [[-1,-1],[-1,0],[1,0],[0,-1],[0,1],[1,1],[-1,1],[1,-1]]
    visited = set()
    q = collections.deque()
    dist = 0
    for ball_p in ball_cnt:
        p = tuple(ball_p.flatten())
        q.append(p)
        visited.add(p)

    while q:
        n = len(q)
        if dist<MIN_DIST_FROM_OTHER_CONTOURS:
            dist += 1
        else:
            break

        for i in range(n):
            p = q.popleft()
            print(f"visited:{visited}")
            print(f"coords:{p}")
            visited.add(p)
            for dir in dirs:
                c = p[0]+dir[0]
                r = p[1]+dir[1]
                print(f"c:{c}")
                print(f"r{r}")
                if c<cols and r<rows and c>0 and r>0 and (c,r) not in visited:
                    print(f"new dir:{(c,r)}")
                    if cnt_mask[r][c]==255:
                        print("FOUND NEARBY POINT")
                        # print(ball_cnt)
                        # print((r,c))
                        cv2.namedWindow("asd", cv2.WINDOW_NORMAL)
                        # cv2.resizeWindow("asd", 800, 600)
                        show = cv2.cvtColor(cnt_mask,cv2.COLOR_GRAY2RGB)
                        cv2.drawContours(show, [ball_cnt], 0, (0,0,255), 3)
                        cv2.circle(show, (r,c), radius=5, color=(0, 255, 0), thickness=-1)  # Green filled circle
                        cv2.imshow("asd",show)
                        key = cv2.waitKey(0) & 0xFF
                        if key == ord('q'):
                            break
                        return True
                    
                    q.append((c,r))
    
    return False

def preprocess_frame(back_sub,frame):
    lower = np.array([0, 0, 100])   # Hue doesn't matter, saturation small, value high
    upper = np.array([179, 120, 255])
    frame = cv2.resize(frame,(DOWNSAMPLE_COLS,DOWNSAMPLE_ROWS),interpolation=cv2.INTER_AREA)
    blur = cv2.GaussianBlur(frame, (7, 7), 1.5)
    fgMask = back_sub.apply(blur)
    _,fgMask = cv2.threshold(fgMask, 130, 255, cv2.THRESH_BINARY)
    
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    white_mask = cv2.inRange(hsv, lower, upper)
    combined = cv2.bitwise_and(fgMask,white_mask)

    fgMask = combined
    return fgMask


def point_side(a, b, p):
    # a, b, p: tuples (x, y)
    val = (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0])
    if val > 0:
        return "left"
    elif val < 0:
        return "right"
    else:
        return "colinear"
    

if __name__ == '__main__':
    # capture = cv2.VideoCapture("myvideos/test60fps.mp4") 
    # capture = cv2.VideoCapture("openData/game_3.mp4")
    capture = cv2.VideoCapture("openData/serve2.mp4")

    output = cv2.imread('images/output_table_flipped.jpg')
    
    fps = capture.get(cv2.CAP_PROP_FPS)
    if fps>=60:
        skip_rate = round(fps/60)
    else:
        skip_rate = 1

    print(fps)
    print(skip_rate)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5)) #ONLY FOR GMG!
    lower = np.array([0, 0, 50])   # Hue doesn't matter, saturation small, value high
    upper = np.array([179, 160, 255])

    ball_history = collections.deque()
    history = 10
    point_started = False
    high_confidence_count = 0
    frames_since_ball = 0
    last_high_conf_area = None
    velocity = None

    ret,prev_frame = capture.read()

    original_rows = prev_frame.shape[0]
    original_cols = prev_frame.shape[1]
    print(original_cols)

    scale_x = original_cols / DOWNSAMPLE_COLS
    scale_y = original_rows / DOWNSAMPLE_ROWS
    
    

    table_quad = find_table.find_table(prev_frame,display=True)

    
   
    table_coords = [ (int(round(x)), int(round(y))) for x, y in  list(table_quad.exterior.coords) ]

    scaled_coords = [(int(x/scale_x), int(y/scale_y)) for x,y in table_coords]
    table_coords=scaled_coords

    print(f"scaled:{scaled_coords}")
    table_quad = shapely.Polygon(scaled_coords)
    if table_quad:
        shapely.prepare(table_quad)
    else:
        print("table not found")

    # for point in scaled_coords:
    #     cv2.circle(frame, point, radius=5, color=(0, 255, 0), thickness=-1)

    # cv2.circle(prev_frame, (400,600), radius=5, color=(0, 255, 0), thickness=-1)


    col_values = [x[0] for x in scaled_coords]
    row_values = [x[1] for x in scaled_coords]
    table_left_end = min(col_values)
    table_right_end = max(col_values)
    table_bottom = max(row_values)

    col_values = sorted(set(col_values))
    inner_left = col_values[1]
    inner_right = col_values[-2]
 
    table_edges = []
    for i in range(len(table_coords)-1):
        if table_coords[i][0] <= table_coords[i+1][0]:
            left =  table_coords[i]
            right = table_coords[i+1]
        else:
            left =  table_coords[i+1]
            right = table_coords[i]

        # table_edges.append((-math.dist(table_coords[i], table_coords[i+1]), table_coords[i],table_coords[i+1]))
        table_edges.append((-math.dist(table_coords[i], table_coords[i+1]), left,right))



    heapq.heapify(table_edges)

    #points for transforming points from side angle to top angle
    src_pts = []
    dst_pts = [(32,60),(32,828),(466,60),(466,828)]

    larg = heapq.heappop(table_edges)



    src_pts.append((larg[1][0],larg[1][1]))
    src_pts.append((larg[2][0],larg[2][1]))
    midpoint1 = ((larg[1][0]+larg[2][0])//2, (larg[1][1]+larg[2][1])//2)
    larg = heapq.heappop(table_edges)
    src_pts.append((larg[1][0],larg[1][1]))
    src_pts.append((larg[2][0],larg[2][1]))
    midpoint2 = ((larg[1][0]+larg[2][0])//2, (larg[1][1]+larg[2][1])//2)
  
    
    # cv2.line(prev_frame, midpoint1, midpoint2, (0,0,255), 1)
    # cv2.imshow("?", prev_frame)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    src_pts = np.float32(src_pts)
    dst_pts = np.float32(dst_pts)

    M, mask = cv2.findHomography(src_pts, dst_pts,0)


    left_edge = heapq.heappop(table_edges)
    print(f"left:{left_edge}")
    right_edge = heapq.heappop(table_edges)
    print(right_edge)
    if (right_edge[1][0]<left_edge[1][0]):
        left_edge, right_edge = right_edge, left_edge



    prev_frame = cv2.resize(prev_frame,(DOWNSAMPLE_COLS,DOWNSAMPLE_ROWS),interpolation=cv2.INTER_AREA)
    prev_frame = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)


    frame_count = -1
    bounces_this_point = []
    bounces = []
    while True:
        frame_count += 1

        ret, frame = capture.read()
        if frame is None:
            break

        if frame_count % skip_rate != 0:
            print("skipped")
            continue

        

        start_time = time.time()

        
    
        frame = cv2.resize(frame,(DOWNSAMPLE_COLS,DOWNSAMPLE_ROWS),interpolation=cv2.INTER_AREA)


        # gray_frame = frame.copy()
        # gray_frame = cv2.cvtColor(gray_frame, cv2.COLOR_BGR2GRAY)
        # frame_diff = cv2.subtract(gray_frame,prev_frame)
        # prev_frame = gray_frame

        # frame_diff = cv2.GaussianBlur(frame_diff, (7, 7), 1.5)
        # ret,diff_mask = cv2.threshold(frame_diff,30,255,cv2.THRESH_BINARY)
        

        blur = cv2.GaussianBlur(frame, (3, 3), 1.5)
        # blur = frame



        fgMask = backSub.apply(blur)
        _,fgMask = cv2.threshold(fgMask, 130, 255, cv2.THRESH_BINARY)

        # combined2 = cv2.bitwise_and(fgMask, diff_mask)

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        white_mask = cv2.inRange(hsv, lower, upper)
        combined = cv2.bitwise_and(fgMask,white_mask)
        fgMask = combined
        
        # cv2.imshow("backsub",fgMask)
        # cv2.imshow("white",white_mask)
        # cv2.imshow("frame_diff", frame_diff)
        # cv2.imshow("diff_mask", diff_mask)
        # cv2.imshow("combined",combined)
        # cv2.imshow("combined2",combined2)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        if not point_started:
            ball_contour = get_ball_contour(fgMask)

        else:
            ball_contour = get_ball_during_point(fgMask, ball_history[-1], table_left_end, table_right_end,roi_topleft,roi_botright,last_high_conf_area)
        

        if ball_contour is not None:
            area = cv2.contourArea(ball_contour)

            if last_high_conf_area is not None:
                print(f"area ratio: {area/last_high_conf_area}")
            # print(f"area:{area}")
            radius = math.sqrt(area / math.pi)
            print(f"r:{radius}")
            perimeter = cv2.arcLength(ball_contour, True)
            circularity = 4 * math.pi * area / (perimeter ** 2)
            print(f"circularity:{circularity}")



            # ball_pos = get_cnt_centroid(ball_contour)
            ball_pos = get_cnt_bottom(ball_contour)
            print(f"ball pos:{ball_pos}")




            #bounce detection
            prev_velocity = None
            if velocity is not None:
                prev_velocity = velocity
            if len(ball_history)>0:
                velocity = (ball_pos[0] - ball_history[-1].position[0], ball_pos[1] - ball_history[-1].position[1])
                print(f"velocity:{velocity}")
            
            print(f"frames since ball:{frames_since_ball}")
            if point_started and frames_since_ball == 0 and prev_velocity is not None and prev_velocity[1]>0 and velocity[1]<0 and table_quad.contains(shapely.Point(ball_history[-1].position)):
                print("BOUNCE")
                cv2.circle(frame, ball_history[-1].position, radius=2, color=(0, 255, 0), thickness=-1)
                transformed_pos = cv2.perspectiveTransform(np.float32([ball_history[-1].position]).reshape(-1,1,2),M)
                bounces_this_point.append((int(transformed_pos[0][0][0]),int(transformed_pos[0][0][1])))
                print(transformed_pos)
                print(transformed_pos[0][0])
                cv2.circle(output, (int(transformed_pos[0][0][0]),int(transformed_pos[0][0][1])), radius=2, color=(0, 255, 0), thickness=-1)
                cv2.imshow("output",output)
                cv2.waitKey(0)
                cv2.destroyAllWindows()




            print(point_side(midpoint2,midpoint1, ball_pos))
            if (table_quad.contains(shapely.Point(ball_pos))):
                print("ball is on the table!")

            

            ball_history.append(BallCandidate(radius,circularity,ball_pos))
            if len(ball_history)>history:
                ball_history.popleft()

            if point_started:
                # frames_since_ball = 0
                roi_topleft = (ball_pos[0]-TRACKING_WINDOW_WIDTH,ball_pos[1]-TRACKING_WINDOW_HEIGHT)
                roi_botright = (ball_pos[0]+TRACKING_WINDOW_WIDTH,table_bottom)

            
            if len(ball_history)>1 and frames_since_ball<10:
                prev_pos = ball_history[-2].position
                displacement_vector = tuple(np.subtract(ball_pos, prev_pos))
            
                #there could be multiple circular objects in the frame etc. this should be improved: makes sure detections are near each other?
                #the change in x is completely arbitrary... do that differently

                near_vertical = (vector_length(displacement_vector) < 5) or (displacement_vector[1]!=0 and abs(math.atan(displacement_vector[0]/displacement_vector[1])) < math.pi/6)

                if circularity > 0.85 and math.dist(ball_pos, prev_pos) < MAX_DIST_FROM_PREVIOUS_POS and near_vertical and (ball_pos[0]<=inner_left or ball_pos[0]>=inner_right):
                    print(f"displacement:{displacement_vector}")
                    frames_since_ball = 0
                    high_confidence_count += 1
                    print(f"conf count:{high_confidence_count}")


            frames_since_ball = 0 #could this lead to issues? not sure   

        else:
            frames_since_ball += 1
        

        if high_confidence_count >=3 and not point_started:
            print("point started!")
            bounces_this_point.clear()
            point_started = True
            high_confidence_count = 0
            last_high_conf_area = area
            if ball_pos[0] < DOWNSAMPLE_COLS//2:
                roi_topleft = (ball_pos[0]-30,ball_pos[1]-50)
                roi_botright = (ball_pos[0]+TRACKING_WINDOW_WIDTH,table_bottom)
            else:
                roi_topleft = (ball_pos[0]-TRACKING_WINDOW_WIDTH,ball_pos[1]-50)
                roi_botright = (ball_pos[0]+30,table_bottom)


        if frames_since_ball>10:
            high_confidence_count = 0   

        if frames_since_ball > fps:
            frames_since_ball = 0
            last_high_conf_area = None
            ball_history.clear()

            if point_started:
                print("point over")
                point_started = False
                if len(bounces_this_point)>0:
                    print("proper point, added to stats")
                    bounces.append(bounces_this_point)
                
        





        fgMask = cv2.cvtColor(fgMask,cv2.COLOR_GRAY2RGB)

        end_time = time.time()
        print(f"time per frame:{end_time-start_time}s")


        display = False
        if display:
            for point in scaled_coords:
                cv2.circle(frame, point, radius=1, color=(0, 255, 0), thickness=-1)

            if point_started:
                cv2.rectangle(fgMask, roi_topleft,roi_botright, (0,255,0),3)


            cv2.drawContours(frame, ball_contour, -1, (0, 0, 255), 1)
            cv2.drawContours(fgMask, ball_contour, -1, (0, 0, 255), 4)
        
            cv2.namedWindow("Frame difference", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("Frame difference", 800, 600)
            cv2.imshow('Frame difference', frame)
            cv2.namedWindow("mask", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("mask", 800, 600)
            cv2.imshow('mask', fgMask)


            key = cv2.waitKey(0) & 0xFF
            if key == ord('q'):
                break

    capture.release()
    cv2.destroyAllWindows()


    stats.calculate(bounces)