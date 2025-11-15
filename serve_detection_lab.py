import collections
import heapq
import json
import math
import time
import cProfile
import pstats

import cv2
import numpy as np
import shapely

# import find_table
import fast_find_table
import stats


class BallCandidate:
    def __init__(self, radius: float, circularity: float, position: tuple, frame,contour,color,area):
        self.radius = radius
        self.circularity = circularity
        self.position = position  # position should be a tuple (x, y)
        self.frame = frame
        self.contour = contour
        self.color = color
        self.area = area

    def __repr__(self):
        return (f"BallCandidate(radius={self.radius}, "
                f"circularity={self.circularity}, "
                f"position={self.position}, "
                f"frame={self.frame})")




# DOWNSAMPLE_ROWS = 360
# DOWNSAMPLE_COLS = 640

DOWNSAMPLE_ROWS = 540
DOWNSAMPLE_COLS = 960

#MIN_DIST_FROM_OTHER_CONTOURS = DOWNSAMPLE_COLS //15 
#OVERLAPPING_CONTOUR_DIST = 10 /2
MIN_RADIUS = 0 # this is a bit scuffed...
MAX_RADIUS = 8
MIN_TOSS_AREA = 5
MIN_TOSS_CIRCULARITY = 0.8
MAX_DIST_FROM_PREVIOUS_POS  = DOWNSAMPLE_COLS // 20
TRACKING_WINDOW_HEIGHT = DOWNSAMPLE_ROWS // 4
TRACKING_WINDOW_WIDTH = DOWNSAMPLE_COLS // 4
MAX_TRACKING_DIST = math.sqrt(TRACKING_WINDOW_HEIGHT**2 + TRACKING_WINDOW_WIDTH**2)
MAX_COLOR_DIFF = 80

MAX_AREA_RATIO = 15

#represents the predicted change in X coord of the ball when it hasn't been detected for at least 1 frame. percentage of table length.  
PREDICTED_POSITION_RATIO = 0.1


def vector_length(v):
    return math.sqrt(sum(coord ** 2 for coord in v))


def get_ball_candidates(frame_mask, strict_table_quad, fgMask, serve_like_events,table_bottom):
    #detecting balls during the toss, which are supposed to be almost perfectly circular. also looks for ball candidates on the table after the toss
    contours, _ = cv2.findContours(frame_mask, cv2.RETR_EXTERNAL     , cv2.CHAIN_APPROX_NONE)
    good_contours = []

    for i,cnt in enumerate(contours):
        area = cv2.contourArea(cnt)
        radius = math.sqrt(area / math.pi)
        perimeter = cv2.arcLength(cnt, True)
        if perimeter==0 or area < MIN_TOSS_AREA:
            continue
        
        cnt_pos = get_cnt_centroid(cnt)
        circularity = 4 * math.pi * area / (perimeter ** 2)
        if radius >= MAX_RADIUS or radius<MIN_RADIUS or circularity<MIN_TOSS_CIRCULARITY:
            continue

       
        good_contours.append(cnt)

    
    contours_inside_table = []
    if len(serve_like_events)>0:
    
        contours, _ = cv2.findContours(fgMask, cv2.RETR_EXTERNAL     , cv2.CHAIN_APPROX_NONE)
        
        for i,cnt in enumerate(contours):
            area = cv2.contourArea(cnt)
            perimeter = cv2.arcLength(cnt, True)
            if perimeter==0 or area < 10:
                continue
            
            cnt_pos = get_cnt_centroid(cnt)
            bot_pos = get_cnt_bottom(cnt)
            if cnt_pos[1]<table_bottom and (strict_table_quad.contains(shapely.Point(bot_pos)) or strict_table_quad.contains(shapely.Point(cnt_pos))):
                contours_inside_table.append(cnt)
                

    return good_contours, contours_inside_table


def get_ball_during_point(frame_mask,hsv,ball_history,left,right,roi_l,roi_r,toss_ball,predicted_positions,possible_x_range):
    #tracking the ball over the table based on appearance and predicted position
    if len(ball_history)>0:
        prev_ball = ball_history[-1]
        prev_pos = prev_ball.position
        prev_cnt = prev_ball.contour
        prev_color = prev_ball.color
        prev_area = prev_ball.area

    prev_velocity = None
    if len(ball_history)>1:
        prev_velocity = tuple(np.subtract(ball_history[-1].position, ball_history[-2].position))

    roi_topleft = roi_l
    roi_botright = roi_r

    # roi = frame_mask[roi_topleft[1]:roi_botright[1], roi_topleft[0]:roi_botright[0]]
    # roi_contours, _ = cv2.findContours(roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE, offset=roi_topleft)

    #TRYING THIS. GETTING CONTOURS ONLY FROM ROI GIVES CUT-OFF BLOBS FROM ARMS ETC
    roi = frame_mask
    roi_contours, _ = cv2.findContours(roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    
    contours = roi_contours
    best_contour = None
    shortest_dist = float('inf')
    ball_area = toss_ball.area
    toss_color = toss_ball.color

    for i,cnt in enumerate(contours):
        pos = get_cnt_centroid(cnt)
        color = get_cnt_color2(cnt,hsv)
        
        #only check "above" the table, should reduce FPs by a lot
        if not (pos[0]>=left and pos[0]<=right):
            continue

        if not (pos[0]>=roi_topleft[0] and pos[0]<=roi_botright[0] and pos[1]>=roi_topleft[1] and pos[1]<=roi_botright[1]):
            continue
        
        area = cv2.contourArea(cnt)
        radius = math.sqrt(area / math.pi)
        perimeter = cv2.arcLength(cnt, True)
        if perimeter == 0 or area == 0:
            continue
        
        if pos[0]==prev_pos[0]:
            print("excluded,same pos")
            continue

        if not (pos[0]>=possible_x_range[0] and pos[0]<=possible_x_range[1]):
            print("excluded,wrong direction")
            continue

        if area < ball_area/3 or area>ball_area*10: 
            print(f"excluded candidate:{area/ball_area} {radius} ") 
            continue

        #print(f"candidate:{area/ball_area} {radius} {circularity}") 
        # ellipse = cv2.fitEllipse(cnt)
        # center, axes, angle = ellipse
        #print(f"Center: {center}")
        #print(f"Axes: {axes} (Major, Minor)")
       # print(f"Angle: {angle} degrees")

        velocity = tuple(np.subtract(pos, ball_history[-1].position))
        if prev_velocity is None or (velocity[0]>0 and prev_velocity[0]>0) or (velocity[0]<0 and prev_velocity[0]<0):
            min_area = prev_area * 0.2
            max_area = prev_area * 5
            max_color_diff = MAX_COLOR_DIFF 
            max_dist = (right-left) / 6
        else:
            min_area = prev_area * 0.2
            max_area = prev_area * 10
            max_color_diff = MAX_COLOR_DIFF 
            max_dist = (right-left) / 6

        if area < min_area or area>max_area: 
            print(f"excluded candidate:{area/prev_area} ") 
            continue
            
            
        if math.dist(color,prev_color)>max_color_diff: #or math.dist(color,toss_color)>MAX_COLOR_DIFF:          
            print(f"excluded due to color:{color}")
            print(get_cnt_color(cnt,hsv))
            continue


        dist = float('inf')
        for pred_pos in predicted_positions:
            dist = min(dist, math.dist(pos,pred_pos))
        
        if dist>max_dist:
            print("excluded,too far")
            continue
        
        if dist < shortest_dist:
            shortest_dist = dist
            best_contour = cnt
    return best_contour


def recover_ball(frame_mask,toss_ball,strict_table_quad, ball_history,hsv):
    contours, _ = cv2.findContours(frame_mask, cv2.RETR_EXTERNAL     , cv2.CHAIN_APPROX_NONE)
    ball_area = toss_ball.area

    for i,cnt in enumerate(contours):
        area = cv2.contourArea(cnt)
        perimeter = cv2.arcLength(cnt, True)
        if perimeter==0 or area < 10:
            continue
        
        if area < ball_area/3 or area>ball_area*10: 
            print(f"excluded candidate:{area/ball_area}") 
            continue
        
        color = get_cnt_color2(cnt,hsv)
        prev_color = ball_history[-1].color

        if math.dist(color,prev_color)>MAX_COLOR_DIFF: #or math.dist(color,toss_color)>MAX_COLOR_DIFF:          
            continue

        cnt_pos = get_cnt_centroid(cnt)
        bot_pos = get_cnt_bottom(cnt)

        if (strict_table_quad.contains(shapely.Point(bot_pos)) or strict_table_quad.contains(shapely.Point(cnt_pos))):
            return cnt
        

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

def get_cnt_color(cnt, hsv):
    mask = np.zeros(hsv.shape[:2], np.uint8)
    cv2.drawContours(mask, cnt, -1, 255, -1)
    mean = cv2.mean(hsv, mask=mask)
    return mean

def get_cnt_color2(cnt,frame):
    dirs = [[0,-1],[0,1],[-1,0],[1,0],[0,0]]

    center = get_cnt_centroid(cnt)
    pixel_values = []
    for dir in dirs:
        pos_x = center[0]+dir[0]
        pos_y = center[1]+dir[1]
        if 0<=pos_x<DOWNSAMPLE_COLS and 0<=pos_y<DOWNSAMPLE_ROWS:
            pixel_values.append(frame[pos_y][pos_x])


    arr = np.array(pixel_values)
    means = np.mean(arr, axis=0)
    return tuple(means.astype(int))

def get_cnt_median_color(cnt,frame):
    dirs = [[0,-1],[0,1],[-1,0],[1,0],[0,0]]
    center = get_cnt_centroid(cnt)
    pixel_values = []
    for dir in dirs:
        pixel_values.append(frame[center[1]+dir[1]][center[0]+dir[0]])
    arr = np.array(pixel_values)
    medians = np.median(arr, axis=0)
    return tuple(medians.astype(int))




def point_side(a, b, p):
    # a, b, p: tuples (x, y)
    #put coliniear case to the left side, just to make this easier
    val = (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0])
    if val >= 0:
        return "Left"
    else:
        return "Right"
    
    

def create_point(point_id, set_number, server, receiver, winner, start, end, bounces,serve_bounce):
    return {
        "point_id": point_id,
        "set_number": set_number,
        "rally_length": len(bounces)-1,
        "server": server,
        "receiver": receiver,
        "winner": winner,
        "point_won_on_serve": winner == server,
        "frame_start": start,
        "frame_end": end,
        "bounces": bounces,
        "serve_bounce":serve_bounce,
        "winning_bounce":bounces[-1]
    }

def transformed_bounce_side(pos):
    table_midpoint = 445
    if pos[1]<table_midpoint: 
        return "Left"
    else:
        return "Right"

def is_point_over(frames_since_ball, last_bounce_frame, curr_frame, real_fps,analysis_fps, bounces,ball_history):
    #if ball tracking gets very robust, maybe track the ball in the whole image, not just over the table and look at bounces on the floor?

    if frames_since_ball > analysis_fps:
        print(f"{analysis_fps} frames since the last ball detection, point over")
        return True
    #NEEDS TUNING
    if (curr_frame - last_bounce_frame) > 3*real_fps:
        print(f"{3*real_fps} frames since the last bounce, point over")
        return True

    if len(ball_history) < 5 and frames_since_ball > 1:
        print("ending point early, probably incorrect serve detection")
        return True
    #NEEDS TUNING: USE 3 BOUNCES, OR FIX BOUNCE DETECTION...
    # if len(bounces)>1 and transformed_bounce_side(bounces[-1][0])==transformed_bounce_side(bounces[-2][0]) and bounces[-1][1]-bounces[-2][1] < real_fps/2:
    #     print("the ball bounced on the same side of the ball in a short timespan, point over")
    #     return True
    
    return False

def remove_large_blobs(mask, max_area=150, kernel_size=10):
    """
    Removes large blobs and small blobs that are close enough to be connected by morphological closing.

    Parameters:
        mask (np.ndarray): Binary input mask.
        max_area (int): Maximum area to keep blobs.
        kernel_size (int): Size of the morphological kernel.

    Returns:
        np.ndarray: Cleaned binary mask.
    """
    # Ensure binary
    # mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)[1]

    # Morphological closing to connect close blobs
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))


    #closed_mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    closed_mask = mask

    # Remove large blobs from closed mask
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(closed_mask, connectivity=8)
    cleaned = np.zeros_like(mask)

    for i in range(1, num_labels):  # Skip background
        area = stats[i, cv2.CC_STAT_AREA]
        if area <= max_area:
            cleaned[labels == i] = 255

    return cleaned

def get_roi_bounds(ball_pos, table_left_end, table_right_end, table_bottom):
    if ball_pos[0]-TRACKING_WINDOW_WIDTH//2 < table_left_end:
        roi_topleft = (table_left_end,max(0,ball_pos[1]-int(TRACKING_WINDOW_HEIGHT*1.5)))
        roi_botright = (table_left_end + TRACKING_WINDOW_WIDTH,table_bottom)
    
    elif ball_pos[0]+TRACKING_WINDOW_WIDTH//2 > table_right_end:
        roi_botright = (table_right_end,table_bottom)
        roi_topleft = (table_right_end - TRACKING_WINDOW_WIDTH, max(0,ball_pos[1]-int(TRACKING_WINDOW_HEIGHT*1.5)))

    else:
        roi_topleft = (ball_pos[0]-TRACKING_WINDOW_WIDTH//2,max(0,ball_pos[1]-int(TRACKING_WINDOW_HEIGHT*1.5)))
        roi_botright = (ball_pos[0]+TRACKING_WINDOW_WIDTH//2,table_bottom)

    return roi_topleft, roi_botright

def is_bounce(ball_history, table_quad,skip_rate):
    bounce_pos = get_cnt_bottom(ball_history[-2].contour)
    if len(ball_history)<3 or not table_quad.contains(shapely.Point(bounce_pos)):
        if not table_quad.contains(shapely.Point(bounce_pos)):
            print("bounce outside table")

        return False
    
    t1 = ball_history[-1].frame-ball_history[-2].frame
    t2 = ball_history[-2].frame-ball_history[-3].frame
    vel2 = tuple(np.subtract(ball_history[-2].position, ball_history[-3].position))
    vel3 = tuple(np.subtract(ball_history[-1].position, ball_history[-2].position))
   
    if t1==t2==skip_rate and vel2[1]>0 and vel3[1]<0:
        return True
    
    if len(ball_history)>3:

        vel1 = tuple(np.subtract(ball_history[-3].position, ball_history[-4].position))
        t3 = ball_history[-3].frame-ball_history[-4].frame
        if (t1==t2==t3==skip_rate and vel1[1]>0 and vel2[1]==0 and vel3[1]<0):
            return True
        
    return False

def is_possible_distortion(ball_radius, contour):
    pass

def get_new_predicted_positions(curr_frame,ball_history,server,left_serve_pos,right_serve_pos,table_length,skip_rate):
    if len(ball_history)==0:
        if server=="Left":
            return [left_serve_pos]
        else:
            return [right_serve_pos]
    
    elif len(ball_history)==1:
        if server=="Left":
            predicted_velocity = (0.05*table_length,0)
        else:
            predicted_velocity = (-0.05*table_length,0)
        
        return [tuple(np.add(np.array(ball_history[-1].position), np.array(predicted_velocity)).tolist())]
    


    else:
        if ball_history[-1].frame == curr_frame and (ball_history[-1].frame - ball_history[-2].frame) == skip_rate:
            velocity = tuple(np.subtract(ball_history[-1].position, ball_history[-2].position))
            pred1 = tuple(np.add(np.array(ball_history[-1].position), np.array(velocity)).tolist())
            pred2 = tuple(np.subtract(np.array(ball_history[-1].position), np.array(velocity)).tolist())
        else:
            predicted_velocity = (PREDICTED_POSITION_RATIO*table_length,0)
            pred1 = tuple(np.add(np.array(ball_history[-1].position), np.array(predicted_velocity)).tolist())
            predicted_velocity = (-PREDICTED_POSITION_RATIO*table_length,0)
            pred2 = tuple(np.add(np.array(ball_history[-1].position), np.array(predicted_velocity)).tolist())
        return [pred1,pred2]

def get_next_data(iterator):
    try:
        item = next(iterator)
        frame_num = int(item[0])
        x = int(item[1]['x'])
        y = int(item[1]['y'])
        return frame_num, x, y
    except StopIteration:
        return None

def is_table_view(frame, table_mask):
    edges = cv2.Canny(frame,80,200)
    common  = cv2.bitwise_and(edges,table_mask)
    overlap_percentage = cv2.countNonZero(common)/cv2.countNonZero(table_mask)
    # print(overlap_percentage)
    if  overlap_percentage > 0.15:
        return True
    else:
        return False


def get_white_threshold(frame,mask):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    v_channel = hsv[:, :, 2]
    s_channel = hsv[:, :, 1]

    
    only_moving_parts = cv2.bitwise_and(hsv,hsv, mask=mask)
    # toshow = cv2.bitwise_and(frame,frame, mask=mask)
    # hsv_moving =  cv2.cvtColor(only_moving_parts, cv2.COLOR_BGR2HSV)
    # cv2.imshow("hehe",toshow)

    ve_channel = only_moving_parts[:, :, 2]
    se_channel = only_moving_parts[:, :, 1]

    

    # Flatten the V channel and filter out zeros
    ve_values = ve_channel.flatten()
    ve_nonzero = ve_values[ve_values > 0]

    se_values = se_channel.flatten()
    se_nonzero = se_values[se_values > 0]
    
    if len(ve_nonzero)>0 and len(se_nonzero)>0:
        ve_thresh = np.percentile(ve_nonzero, 70)
        se_thresh = np.percentile(se_nonzero, 50)
        
    
        # print('MOVING THRESHOLDS')
        # print(ve_thresh)
        # print(se_thresh)
        binary_map = (v_channel >= ve_thresh) & (s_channel <= se_thresh)
        # Convert boolean mask to uint8 (0 or 1)
        
        binary_image = (binary_map.astype(np.uint8)) * 255
        binary_image = cv2.bitwise_and(binary_image,mask)
        return ve_thresh, se_thresh
    else:
        return 0, 255
        # cv2.imshow("white mask", binary_image)

def apply_white_filter(frame, mask, ve_thresh, se_thresh):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # binary_map = (v_channel >= ve_thresh) & (s_channel <= se_thresh)
    binary_map = cv2.inRange(hsv,(0,0,ve_thresh),(179,se_thresh,255))
    # Convert boolean mask to uint8 (0 or 1)
    
    # binary_image = (binary_map.astype(np.uint8)) * 255
    binary_image = cv2.bitwise_and(binary_map,mask)

    return binary_image


def update_serve_candidates(serve_candidates, ball_candidates, frame_count,table_quad,serve_events,frame, inner_left, inner_right,table_bottom,fps):
    max_dist = 20
    max_frames_between = 20
    serve_candidates = [cand for cand in serve_candidates if ((frame_count - cand[-1].frame) <= max_frames_between)and len(cand)<2]

    for ball_cand in ball_candidates:
        ball_cand_pos  = get_cnt_centroid(ball_cand)
        ball_cand_color = get_cnt_color2(ball_cand,frame)
        ball_cand_area = cv2.contourArea(ball_cand)
        found = False
        if not table_quad.contains(shapely.Point(ball_cand_pos)) and (ball_cand_pos[0]<=inner_left or ball_cand_pos[0]>=inner_right) and ball_cand_pos[1]<table_bottom:
            for serve_cand in serve_candidates:
                if 0 < math.dist(ball_cand_pos, serve_cand[-1].position)<max_dist and (frame_count-serve_cand[-1].frame)>0:
                    
                    serve_cand.append(BallCandidate(1,1,ball_cand_pos, frame_count,ball_cand,ball_cand_color,ball_cand_area))

                    if len(serve_cand)>=2:
                        serve_events.append(BallCandidate(1,1,ball_cand_pos, frame_count,ball_cand,ball_cand_color,ball_cand_area))
                        print("PROBABLY TOSS!!")
                        print(ball_cand_area)
                        print(ball_cand_color)

                        # cv2.drawContours(frame, ball_cand, -1, (0, 0, 255), 1)
                        # cv2.namedWindow("Frame difference", cv2.WINDOW_NORMAL)
                        # cv2.resizeWindow("Frame difference", 800, 600)
                        # cv2.imshow('Frame difference', frame)

                        
                        # key = cv2.waitKey(0) & 0xFF
                        # if key == ord('q'):
                        #     break

                    found = True
                    break

            if not found:
                serve_candidates.append([BallCandidate(1,1,ball_cand_pos, frame_count,ball_cand,ball_cand_color,ball_cand_area)])
    
    

    serve_events = [event for event in serve_events if frame_count - event.frame < fps*1.2]
    print(serve_events)
    return serve_candidates

def point_is_starting(mask, strict_table_quad, serve_like_events, contours_inside_table,frame_count,real_fps,hsv, midpoint1,midpoint2):
    if len(serve_like_events)==0 or len(contours_inside_table)==0:
        return False, None, None
    
    
    for cnt_on_table in contours_inside_table:
        pos = get_cnt_centroid(cnt_on_table)
        color = get_cnt_color2(cnt_on_table,hsv)
        # colorr = get_cnt_color(cnt_on_table,hsv)
        # median_color = get_cnt_median_color(cnt_on_table,hsv)
        inside_side = point_side(midpoint2,midpoint1, pos)
        inside_area = cv2.contourArea(cnt_on_table)

        print("contour on table:")
        print(color)
       

        print(inside_area)
        print(f"Side:{ point_side(midpoint2,midpoint1, pos)}")

        for serve_event in serve_like_events:
            serve_pos = get_cnt_centroid(serve_event.contour)
            serve_area = serve_event.area
            # print(point_side(midpoint2,midpoint1, serve_pos))
            # print(serve_event)


            if (frame_count - serve_event.frame < 1.1*real_fps  
                and 0.8<inside_area/serve_area<5 and math.dist(color,serve_event.color)<MAX_COLOR_DIFF ): #and point_side(midpoint2,midpoint1, serve_pos)==inside_side:
                print("serve color diff")
                print(serve_event.color)
                print(color)
                print("area diff")
                print(serve_area)
                print(inside_area)

                serve_like_events.remove(serve_event) #it shouldn't be reused. if early exit, don't repeat 

                return True, serve_event, cnt_on_table

    
    return False, None,  None

def get_possible_x_range(ball_history,server, midpoint1,midpoint2,table_left_end,table_right_end, frames_since_ball):
    if len(ball_history)==0 or frames_since_ball>=5:
        return [0,DOWNSAMPLE_COLS]
    prev_pos = ball_history[-1].position
    side = point_side(midpoint2,midpoint1,prev_pos)
    
    
    if len(ball_history)==1: #or 1?
        if server=="Left":
            return [prev_pos[0],DOWNSAMPLE_COLS]
        else:
            return [0,prev_pos[0]]
    else:
        velocity = tuple(np.subtract(ball_history[-1].position, ball_history[-2].position))
        # print(f"velocity:{velocity}")
        if side=="Left" and velocity[0]>0:
            return [prev_pos[0],DOWNSAMPLE_COLS]
        elif side=="Right" and velocity[0]<0:
            return [0,prev_pos[0]]
        
        #this is specific to when only tracking inside table bounds

        if prev_pos[0]+velocity[0]<table_left_end:
            return [prev_pos[0],DOWNSAMPLE_COLS]
        elif prev_pos[0]+velocity[0]>table_right_end:
            return [0,prev_pos[0]]
    
    return [0,DOWNSAMPLE_COLS]

def timestamp_to_framecount(filepath,fps):
    # Read timestamps from a text file
    with open(filepath, "r") as f:
        content = f.read().strip()

    # Split timestamps by spaces or newlines
    timestamps = content.replace("\n", " ").split()

    # Convert each timestamp (mm:ss) to frame count
    frame_counts = []
    for t in timestamps:
        minutes, seconds = map(int, t.split(":"))
        total_seconds = minutes * 60 + seconds
        frame_count = total_seconds * fps
        frame_counts.append(int(frame_count))


    return frame_counts




def main(video_path, stop_event=None, metadata_queue = None, progress_callback = None, display=False, eval = False):

    overall_start = time.time()
    # display = True
    # eval = True
    #evaluation stuff
    BALL_POS_PATH = 'openData/game_1/ball_markup.json'
    with open(BALL_POS_PATH) as json_file:
        data = json.load(json_file)


    dict_iter = iter(data.items())
    eval_frame, ball_x, ball_y = get_next_data(dict_iter)
    tp = 0
    fp = 0
    tn = 0
    fn = 0
    serve_tp = 0
    serve_fp = 0

    # capture = cv2.VideoCapture("myvideos/test60fps.mp4") 
    # capture = cv2.VideoCapture("myvideos/random.mkv")
    # capture = cv2.VideoCapture("openData/game_2.mp4")
    # capture = cv2.VideoCapture("openData/serve2.mp4")
    # capture = cv2.VideoCapture("myvideos/wtt2.webm")

    capture = cv2.VideoCapture(video_path)
    output = cv2.imread('images/output_table_flipped.jpg')

    
    fps = capture.get(cv2.CAP_PROP_FPS)
    backSub = cv2.createBackgroundSubtractorMOG2(varThreshold=12, detectShadows=True, history=int(fps)*20)
    print(backSub.getNMixtures())
    print(backSub.getShadowThreshold())
    print(backSub.getDetectShadows())
    print(backSub.getBackgroundRatio())
    print(backSub.getHistory())
    #backSub.setShadowThreshold(0.25)
    # time.sleep(10)
    nr_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # if fps>=60:
    #     skip_rate = round(fps/60)
    # else:
    #     skip_rate = 1

    if fps>=30:
        skip_rate = round(fps/30)
    else:
        skip_rate = 1

    analysis_fps  = round(fps/skip_rate)
    print(fps)
    print(skip_rate)

    
    serve_timestamps = timestamp_to_framecount("openData/game_1/serves.txt", fps)


    ball_history = collections.deque()
    history = 10
    point_started = False
    high_confidence_count = 0
    frames_since_ball = 0
    last_high_conf_area = 10

    frame_count = 0
    ret,prev_frame = capture.read()

    original_rows = prev_frame.shape[0]
    original_cols = prev_frame.shape[1]
    print(original_cols)

    scale_x = original_cols / DOWNSAMPLE_COLS
    scale_y = original_rows / DOWNSAMPLE_ROWS
    

    tries = 0
    table_quad = None
    while table_quad is None and tries<100:
        #for WTT specifically
        if frame_count>50:
            # prev_frame = cv2.resize(prev_frame,(DOWNSAMPLE_COLS,DOWNSAMPLE_ROWS),interpolation=cv2.INTER_AREA)
            #table_quad = find_table.find_table(prev_frame,display=False)
            table_quad = fast_find_table.find_table(prev_frame,display=False)
            tries +=1
        ret,prev_frame = capture.read()
        frame_count += 1
        print(f"FINDING TABLE ATTEMPT NR:{tries}")  
    
    #wtt2
    #table_coords = [(681, 520), (644, 668), (1270, 666), (1232, 521), (681, 520)]

    #game 1 [(249, 673), (1458, 691), (1290, 540), (503, 527), (249, 673)]

    table_coords = [ (int(round(x)), int(round(y))) for x, y in  list(table_quad.exterior.coords) ]
    print(table_coords)
    height, width = prev_frame.shape[:2]
    np_coords = np.int32(table_coords)
    print(np.int32([table_coords]))
    table_mask = np.zeros((height, width), dtype=np.uint8)
    cv2.polylines(table_mask,np.int32([table_coords]), True, 255)
   

    scaled_coords = [(int(x/scale_x), int(y/scale_y)) for x,y in table_coords]
    table_coords=scaled_coords

    print(f"scaled:{scaled_coords}")
    table_quad = shapely.Polygon(scaled_coords)

    strict_table_quad = table_quad.buffer(0, join_style="mitre")
    #this should help include bounces which are right on the boundary of the table
    table_quad = table_quad.buffer(5, join_style="mitre")

    
    if table_quad:
        shapely.prepare(table_quad)
        shapely.prepare(strict_table_quad)
    else:
        print("table not found")


    col_values = [x[0] for x in scaled_coords]
    row_values = [x[1] for x in scaled_coords]
    table_left_end = min(col_values)
    table_right_end = max(col_values)
    table_bottom = max(row_values)
    table_length = table_right_end - table_left_end

    #EXTENDING TRACKING REGION: TESTING!
    table_left_end -= 100
    table_right_end += 100

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
    reverse_M = np.linalg.inv(M)

    left_serve_spot = (250,200)
    right_serve_spot = (250,690)
    transformed_left = cv2.perspectiveTransform(np.float32([left_serve_spot]).reshape(-1,1,2),reverse_M)[0][0]
    transformed_left = (int(transformed_left[0]),int(transformed_left[1]))
    transformed_right = cv2.perspectiveTransform(np.float32([right_serve_spot]).reshape(-1,1,2),reverse_M)[0][0]
    transformed_right = (int(transformed_right[0]),int(transformed_right[1]))
    

    left_edge = heapq.heappop(table_edges)
    print(f"left:{left_edge}")
    right_edge = heapq.heappop(table_edges)
    print(right_edge)
    if (right_edge[1][0]<left_edge[1][0]):
        left_edge, right_edge = right_edge, left_edge



    prev_frame = cv2.resize(prev_frame,(DOWNSAMPLE_COLS,DOWNSAMPLE_ROWS),interpolation=cv2.INTER_AREA)
    print(transformed_left)
    

    prev_frame = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

    table_midpoint = 445
    bounces_this_point = []
    bounces = []
    points_metadata = []
    rally_length = 0
    server = None
    receiver = None
    winner = None
    point_start_frame = None
    last_bounce_frame = None
    predicted_positions = None
    serve_candidates = []
    serve_like_events = []
    possible_x_range = [0,DOWNSAMPLE_COLS]
    serve_bounce = None

    min_value ,max_saturation = None,None

    while True:
        if stop_event is not None and stop_event.is_set():
            print("Task cancelled!")
            return None

        if frame_count % (fps*5) == 0 and progress_callback is not None:
            progress_callback(frame_count, nr_frames)
        
        frame_count += 1
       
        #time1 = time.time()
       
        if frame_count % skip_rate != 0:
            ret = capture.grab()
            continue

        ret, frame = capture.read()
        
        #time2  =time.time()


        # if not is_table_view(frame,table_mask):
        #     continue
        
        #print(f"time to read:{time2-time1}s")
        if frame is None:
            break
        
        #game 4, complete set
        # if frame_count<nr_frames*0.43:
        #     continue
        
        # if frame_count < nr_frames*0.24:
        #     continue

        # game 1, complete set
        if frame_count<nr_frames*0.08:
            continue
        
        if frame_count>nr_frames*0.57:
            break


        #game 3, first set
        # if frame_count>nr_frames*0.5:
        #     break
        
        #game 3, second set
        # if frame_count<nr_frames*0.5:
        #     continue


        # if frame_count<nr_frames*0.57:
        #     continue
        


        

        
        

        start_time = time.time()
        frame = cv2.resize(frame,(DOWNSAMPLE_COLS,DOWNSAMPLE_ROWS),interpolation=cv2.INTER_AREA)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)

        # frame = frame[0:original_rows, 0:int(original_cols//2)]
        # frame = frame[0:original_rows, int(original_cols//2):original_cols]

        #blur = cv2.GaussianBlur(frame, (3, 3), 1.5)
        # blur = cv2.medianBlur(frame, 5)
        #blur = cv2.GaussianBlur(frame, (5, 5), 1.5)
        #blur = cv2.GaussianBlur(frame, (7, 7), 1.5)

        blur = frame
        fgMask = backSub.apply(blur, learningRate =0.0005)
        _,fgMask = cv2.threshold(fgMask, 130, 255, cv2.THRESH_BINARY)

        
            

        if point_started:
            fgMask = cv2.medianBlur(fgMask, 5)
            # fgMask = remove_large_blobs(fgMask,max_area=last_high_conf_area*15)
            
        
            combined = fgMask
            # combined = apply_white_filter(frame, fgMask,min_value,max_saturation)

        else:
            if min_value is None or frame_count%fps==0:
                min_value, max_saturation = get_white_threshold(frame,fgMask) 
            combined = apply_white_filter(frame,fgMask,min_value,max_saturation)
            
            # if len(serve_like_events)>0:
            #     fgMask = cv2.medianBlur(fgMask, 5)
            #     fgMask = remove_large_blobs(fgMask,max_area=last_high_conf_area*15)
        
       
      

        
        if not point_started:
            ball_candidates, contours_inside_table = get_ball_candidates(combined, strict_table_quad, fgMask, serve_like_events, table_bottom)
            serve_candidates = update_serve_candidates(serve_candidates,ball_candidates,frame_count,strict_table_quad,serve_like_events,hsv, inner_left,inner_right,table_bottom,fps)
        else:
            if frames_since_ball < 5:
                ball_contour = get_ball_during_point(combined, hsv,ball_history, table_left_end, table_right_end,roi_topleft,roi_botright,toss_ball, predicted_positions,possible_x_range)  # noqa: F821
            else:
                print("RECOVERING")
                ball_contour = recover_ball(combined, toss_ball, strict_table_quad,ball_history, hsv)  # noqa: F821

            # ball_contour = get_ball_during_point(combined, hsv,ball_history, table_left_end, table_right_end,roi_topleft,roi_botright,toss_ball, predicted_positions,possible_x_range)  # noqa: F821


        # ball_contour = ball_candidates


        
        
        mistake = False
        if eval:
            was_fp = False
            #evaluation
            try:
                while eval_frame<frame_count:
                    eval_frame, ball_x, ball_y = get_next_data(dict_iter)
            except:
                break
            if frame_count==eval_frame:
        
                true_pos = (int(ball_x/scale_x), int(ball_y/scale_y))
                # print(f"TRUE POSITION:{true_pos}")

                above_table = table_left_end <= true_pos[0] <=table_right_end
                if point_started:
                    if ball_contour is None:
                        if (ball_x,ball_y) == (-1,-1) or not above_table:
                            tn += 1
                        else:
                            fn += 1
                            mistake = True
                            # if not display:
                            #     cv2.namedWindow("Frame difference", cv2.WINDOW_NORMAL)
                            #     cv2.resizeWindow("Frame difference", 800, 600)
                            #     cv2.imshow('Frame difference', frame)
                            #     key = cv2.waitKey(0) & 0xFF
                    else:
                        ball_pos = get_cnt_centroid(ball_contour)
                        if (ball_x,ball_y) != (-1,-1) and above_table and math.dist(true_pos,ball_pos) < 20:
                            tp += 1
                        else:
                            fp += 1
                            was_fp = True
                            print(f"DIST:{math.dist(true_pos,ball_pos)}")
                            print((ball_x,ball_y))
                            print(true_pos)
                            print(ball_pos)


        if point_started:
            if ball_contour is not None:
                # start_time1 = time.time()
                # print(f"COLOR:{get_cnt_color(ball_contour,hsv)}")
                # end_time1 = time.time()
                # print(f"time for color:{end_time1-start_time1}s")

                # start_time1 = time.time()
                print(f"COLOR2:{get_cnt_color2(ball_contour,hsv)}")
                # end_time1 = time.time()
                # print(f"time for color2:{end_time1-start_time1}s")
                # print(f"median color:{get_cnt_median_color(ball_contour,hsv)}")

                area = cv2.contourArea(ball_contour)
                if last_high_conf_area is not None:
                    print(f"area ratio: {area/last_high_conf_area}")
                # print(f"area:{area}")
                radius = math.sqrt(area / math.pi)
                print(f"r:{radius}")
                perimeter = cv2.arcLength(ball_contour, True)
                circularity = 4 * math.pi * area / (perimeter ** 2)
                print(f"circularity:{circularity}")
                ball_pos = get_cnt_centroid(ball_contour)
                print(f"ball pos:{ball_pos}")

                
                ball_color = get_cnt_color2(ball_contour,hsv)

                ball_history.append(BallCandidate(radius,circularity,ball_pos, frame_count,ball_contour,ball_color,area))

                velocity = tuple(np.subtract(ball_history[-1].position, ball_history[-2].position))


                if len(ball_history)>history:
                    ball_history.popleft()

                possible_x_range = get_possible_x_range(ball_history,server,midpoint1,midpoint2,table_left_end,table_right_end,frames_since_ball)
                print(possible_x_range)
                #predicted_positions = get_new_predicted_positions(frame_count,ball_history,server,transformed_left,transformed_right,table_length,skip_rate)

                if is_bounce(ball_history,table_quad,skip_rate):
                    print("BOUNCE")
                    last_bounce_frame = frame_count
                    bounce_pos = get_cnt_bottom(ball_history[-2].contour)
                    bounce_side = point_side(midpoint2,midpoint1, bounce_pos)
                    if (velocity[0]>0 and bounce_side=="Right") or (velocity[0]<0 and bounce_side=="Left") or len(bounces_this_point)==0:
                        cv2.circle(frame, bounce_pos, radius=2, color=(0, 255, 0), thickness=-1)
                        transformed_pos = cv2.perspectiveTransform(np.float32([bounce_pos]).reshape(-1,1,2),M)
                        bounces_this_point.append(((int(transformed_pos[0][0][0]),int(transformed_pos[0][0][1])), frame_count))
                        print(transformed_pos)
                        print(transformed_pos[0][0])
                        if display:
                            cv2.circle(output, (int(transformed_pos[0][0][0]),int(transformed_pos[0][0][1])), radius=2, color=(0, 255, 0), thickness=-1)
                            cv2.imshow("output",output)
                            cv2.waitKey(0)
                            cv2.destroyAllWindows()
                    else:
                        print("BOUNCE IS NOT LEGAL!")
                    
                    if serve_bounce is None and bounce_side != server:
                        serve_bounce = bounces_this_point[-1][0]
                        print(f"serve bounce:{serve_bounce}")
                        

                roi_topleft, roi_botright = get_roi_bounds(ball_pos, table_left_end, table_right_end, table_bottom)
                frames_since_ball = 0
            else:
                frames_since_ball += 1


            predicted_positions = get_new_predicted_positions(frame_count,ball_history,server,transformed_left,transformed_right,table_length,skip_rate)

            if is_point_over(frames_since_ball,last_bounce_frame,frame_count,fps,analysis_fps,bounces_this_point,ball_history):
                print("point over")
                point_started = False
                if len(bounces_this_point)>0:
                    if bounces_this_point[-1][0][1]<table_midpoint: #bounced on the left side for the last time
                        winner = "Right"
                    else:
                        winner = "Left"

                    print("proper point, added to stats")

                    just_bounce_positions = [x[0] for x in bounces_this_point]
                    points_metadata.append(create_point(1,1,server,receiver,winner,point_start_frame,frame_count,just_bounce_positions,serve_bounce))
                    print(points_metadata[-1])
                    bounces.append(bounces_this_point.copy())
                

        else:
            point_starting, toss_ball, cnt_on_table =  point_is_starting(fgMask,strict_table_quad,serve_like_events,contours_inside_table,frame_count,fps,hsv, midpoint1, midpoint2)
            if point_starting:
                #serve detection evaluation
                found_timestamp = False
                for timestamp in serve_timestamps:
                    if  -fps*2 < (frame_count-timestamp) < fps*2:
                        serve_tp += 1
                        found_timestamp = True
                        break
                
                if not found_timestamp:
                    serve_fp +=1
                        

                serve_bounce = None
                ball_contour = cnt_on_table
                ball_color = get_cnt_color2(ball_contour,hsv)
                ball_pos = get_cnt_centroid(cnt_on_table)
                area = cv2.contourArea(cnt_on_table)
                ball_history.clear() #not 100% about this, but it makes sense no? you get occlusion during the serve, position becomes irrelevant after hit
                ball_history.append(BallCandidate(1,1,ball_pos, frame_count,ball_contour,ball_color,area))
                serve_candidates.clear()

                #not clearing, old serve events just get removed constantly anyway
                #serve_like_events.clear()

                toss_pos = get_cnt_centroid(toss_ball.contour)
                toss_area = toss_ball.area

                if point_side(midpoint2,midpoint1, toss_pos)=="Left":
                    server = "Left"
                    receiver = "Right"
                else:
                    server = "Right"
                    receiver = "Left"
                

                possible_x_range = get_possible_x_range(ball_history,server,midpoint1,midpoint2,table_left_end,table_right_end, frames_since_ball)
                predicted_positions = get_new_predicted_positions(frame_count,ball_history,server,transformed_left,transformed_right,table_length,skip_rate)
                point_start_frame = frame_count
                last_bounce_frame = frame_count
                print("point started!")
                bounces_this_point.clear()
                point_started = True
                last_high_conf_area = toss_area
                print(f"HIGH CONF AREA: {toss_area}")
                roi_topleft, roi_botright = get_roi_bounds(ball_pos, table_left_end, table_right_end, table_bottom)
                
                # cv2.drawContours(frame, ball_contour, -1, (0, 0, 255), 1)
                # cv2.namedWindow("Frame difference", cv2.WINDOW_NORMAL)
                # cv2.resizeWindow("Frame difference", 800, 600)
                # cv2.imshow('Frame difference', frame)
                # cv2.namedWindow("mask", cv2.WINDOW_NORMAL)
                # cv2.resizeWindow("mask", 800, 600)
                # cv2.imshow('mask', fgMask)
                
                # key = cv2.waitKey(0) & 0xFF
                # if key == ord('q'):
                #     break
        



        # if point_started and ball_contour is None:
        #     ball_history.clear()


        #if the ball hasn't been seen in 1s, the point should be over?
        if frames_since_ball > analysis_fps:
            #i don't we ever get here anymore, due to is_point_over
            frames_since_ball = 0
            last_high_conf_area = 10
            ball_history.clear()


        if display: #or mistake:
            combined = cv2.cvtColor(combined,cv2.COLOR_GRAY2RGB)
            for point in scaled_coords:
                cv2.circle(frame, point, radius=1, color=(0, 255, 0), thickness=-1)

            if point_started:
                cv2.rectangle(combined, roi_topleft,roi_botright, (0,255,0),3)

                for pos in predicted_positions:
                    # print(pos)
                    cv2.circle(frame, (int(pos[0]),int(pos[1])), radius=1, color=(255, 0, 255), thickness=2)
                
            #     if eval:
            #         cv2.circle(frame, true_pos, radius=1, color=(255, 0, 0), thickness=2)
            if point_started:
                cv2.drawContours(frame, ball_contour, -1, (0, 0, 255), 1)
                cv2.drawContours(combined, ball_contour, -1, (0, 0, 255), 1)
        
            cv2.namedWindow("Frame difference", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("Frame difference", 800, 600)
            cv2.imshow('Frame difference', frame)

            cv2.namedWindow("mask", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("mask", 800, 600)
            cv2.imshow('mask', combined)

            if point_started:
                cv2.namedWindow("playremove", cv2.WINDOW_NORMAL)
                cv2.resizeWindow("playremove", 800, 600)
                cv2.imshow('playremove', fgMask)
       
            key = cv2.waitKey(0) & 0xFF
            if key == ord('q'):
                break

    capture.release()
    cv2.destroyAllWindows()



    # print(points_metadata)
    # stats.get_stats(points_metadata)

    if eval:
        precision = tp/(tp+fp)
        recall = tp/(tp+fn)
        print(f"precision:{precision}")
        print(f"recall:{recall}")
        print(tp)
        print(tn)
        print(fp)
        print(fn)

    print("serve detection accuracy:")
    print(f"tp:{serve_tp}")
    print(f"fp:{serve_fp}")
    overall_end = time.time()
    print(f"processing time:{overall_end-overall_start}")
    # stats.calculate(midpoint1,midpoint2,bounces)
    
    if metadata_queue is not None:
        # print(points_metadata)
        metadata_queue.put(points_metadata)
        if progress_callback is not None:
            progress_callback(frame_count,nr_frames)

    return points_metadata

if __name__ == '__main__':
    # cProfile.run('main()', sort='cumtime')
    path = "openData/game_1.mp4"
    # path = "myvideos/random.mkv"
    profiler = cProfile.Profile()
    profiler.enable()
    asd1 = time.time()
    meta = main(path,display=False,eval=True)
    asd2 = time.time()
    print(asd2-asd1)
    profiler.disable()
    stats = pstats.Stats(profiler).sort_stats('cumtime')
    stats.print_stats(30)

    print(meta)