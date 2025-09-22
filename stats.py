import pandas as pd
import cv2

def point_side(a, b, p):
    # a, b, p: tuples (x, y)
    val = (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0])
    if val > 0:
        return 0 #"left"
    elif val < 0:
        return 1 #"right"
    else:
        return 2 #"colinear"

def calculate(netp1,netp2,bounces):
    print(bounces)
    table_midpoint = 445 #the y coordinate of the net in the output image

    left_points = 0
    right_points = 0
    print(f"number of points:{len(bounces)}")

    for point in bounces:
        print(point[-1])
        # if point_side(netp1,netp2,point[-1])==0:
        #     right_points += 1
        # elif point_side(netp1,netp2,point[-1])==1:
        #     left_points += 1
        if point[-1][1]<table_midpoint: #bounced on the left side for the last time
            right_points += 1
        elif point[-1][1]>table_midpoint:
            left_points += 1
    
    print(f"left won:{left_points}")
    print(f"right won:{right_points}")


# Bucket rally lengths (e.g., 1-3, 4-6, 7+)
def bucket_rally(length):
    if length == 2:
        return "Direct Serve"
    elif length <= 3:
        return "Short"
    elif length <= 6:
        return "Medium"
    else:
        return "Long"
    
def bucket_serves(serve_pos):
    if serve_pos[1] < 156 or serve_pos > 733:
        return "Long"
    elif serve_pos[1] < 252 or serve_pos > 637:
        return "Half Long"
    else:
        return "Short"


def get_stats(points):
    output = cv2.imread('images/output_table_flipped.jpg')
        # Assume list of points is a list of dicts
    df = pd.DataFrame(points)

    # Basic stats per player
    players = df['server'].unique()

    stats = {}

    print(f"nr of points:{df.shape[0]}")

    for player in players:
        total_points_won = df[df['winner'] == player].shape[0]
        total_serves = df[df['server'] == player].shape[0]
        points_won_on_serve = df[(df['server'] == player) & (df['winner'] == player)].shape[0]
        rally_lengths = df[df['winner'] == player]['rally_length']
        winning_bounces = df[df['winner'] == player]['bounces'].str[-1].tolist()
        
        output_copy = output.copy()
        for pos in winning_bounces:
            cv2.circle(output_copy, pos, radius=2, color=(0, 255, 0), thickness=-1)
            
        # cv2.imshow(player,output_copy)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        stats[player] = {
            "points_won": total_points_won,
            "serve_win_rate": points_won_on_serve / total_serves if total_serves > 0 else 0,
            "avg_rally_length_on_wins": rally_lengths.mean() if not rally_lengths.empty else 0,
            "winning_bounces": winning_bounces
        }
        print(player)
        print(stats[player])

    

    df['rally_category'] = df['rally_length'].apply(bucket_rally)

    # Group by rally category and winner
    rally_stats = df.groupby(['winner', 'rally_category']).size().unstack(fill_value=0)

    # Total points per rally category
    total_per_category = df.groupby(['rally_category']).size()

    # Win rate per player per category
    winrate_by_rally = rally_stats.div(total_per_category, axis=1)

    print(rally_stats)
    print(total_per_category)
    print(winrate_by_rally)
