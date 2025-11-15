import pandas as pd
import cv2


def point_side(a, b, p):
    # a, b, p: tuples (x, y)
    val = (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0])
    if val > 0:
        return 0  # "left"
    elif val < 0:
        return 1  # "right"
    else:
        return 2  # "colinear"


def calculate(netp1, netp2, bounces):
    print(bounces)
    table_midpoint = 445  # the y coordinate of the net in the output image

    left_points = 0
    right_points = 0
    print(f"number of points:{len(bounces)}")

    for point in bounces:
        print(point[-1])
        # if point_side(netp1,netp2,point[-1])==0:
        #     right_points += 1
        # elif point_side(netp1,netp2,point[-1])==1:
        #     left_points += 1
        if point[-1][1] < table_midpoint:  # bounced on the left side for the last time
            right_points += 1
        elif point[-1][1] > table_midpoint:
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
    if serve_pos is None:
        return "Weird"
    
    if serve_pos[1] < 156 or serve_pos[1] > 733:
        return "Long"
    elif serve_pos[1] < 252 or serve_pos[1] > 637:
        return "Half Long"
    else:
        return "Short"


def get_stats(points):
    output = cv2.imread("images/output_table_flipped.jpg")
    # Assume list of points is a list of dicts
    df = pd.DataFrame(points)

    print(df)

    # Basic stats per player
    players = df["server"].unique()

    summary_stats = {}

    print(f"nr of points:{df.shape[0]}")

    for player in players:
        total_points_won = df[df["winner"] == player].shape[0]
        total_serves = df[df["server"] == player].shape[0]
        points_won_on_serve = df[
            (df["server"] == player) & (df["winner"] == player)
        ].shape[0]
        points_won_on_opponent_serve = df[
            (df["server"] != player) & (df["winner"] == player)
        ].shape[0]
        rally_lengths = df[df["winner"] == player]["rally_length"]
        winning_bounces = df[df["winner"] == player]["bounces"].str[-1].tolist()

        output_copy = output.copy()
        for pos in winning_bounces:
            cv2.circle(output_copy, pos, radius=2, color=(0, 255, 0), thickness=-1)

        # cv2.imshow(player,output_copy)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        summary_stats[player] = {
            "Points Won": total_points_won,
            "Points won on own serve": points_won_on_serve,
            "Points won on opponent serve": points_won_on_opponent_serve,
            "Serve Win %": points_won_on_serve / total_serves
            if total_serves > 0
            else 0,
            "Avg Rally Length (Win)": rally_lengths.mean()
            if not rally_lengths.empty
            else 0,
            "winning_bounces": winning_bounces,
        }
        print(player)
        print(summary_stats[player])

    df["rally_category"] = df["rally_length"].apply(bucket_rally)

    df["serve_length"] = df["serve_bounce"].apply(bucket_serves)

    print(df)

    # Group by rally category and winner
    rally_stats = df.groupby(["winner", "rally_category"]).size().unstack(fill_value=0)

    # Total points per rally category
    total_per_category = df.groupby(["rally_category"]).size()

    # Win rate per player per category
    winrate_by_rally = rally_stats.div(total_per_category, axis=1)

    # print(rally_stats)
    # print(total_per_category)
    # print(winrate_by_rally)

    print(summary_stats)


    return summary_stats




class Stats:
    def __init__(self,points):
        df = pd.DataFrame(points)
        df["rally_category"] = df["rally_length"].apply(bucket_rally)
        df["serve_length"] = df["serve_bounce"].apply(bucket_serves)
        self.df = df
    
    def get_summary_statistics(self):
        # Basic stats per player
        players = self.df["server"].unique()

        summary_stats = {}

        

        for player in players:
            total_points_won = self.df[self.df["winner"] == player].shape[0]
            total_serves = self.df[self.df["server"] == player].shape[0]
            points_won_on_serve = self.df[
                (self.df["server"] == player) & (self.df["winner"] == player)
            ].shape[0]
            points_won_on_opponent_serve = self.df[
                (self.df["server"] != player) & (self.df["winner"] == player)
            ].shape[0]
            rally_lengths = self.df[self.df["winner"] == player]["rally_length"]
            winning_bounces = self.df[self.df["winner"] == player]["bounces"].str[-1].tolist()

            

            summary_stats[player] = {
                "Points Won": total_points_won,
                "Points won on own serve": points_won_on_serve,
                "Points won on opponent serve": points_won_on_opponent_serve,
                "Serve Win %": points_won_on_serve / total_serves
                if total_serves > 0
                else 0,
                "Avg Rally Length (Win)": rally_lengths.mean()
                if not rally_lengths.empty
                else 0,
                "winning_bounces": winning_bounces,
            }
            print(player)
            print(summary_stats[player])

        self.df["rally_category"] = self.df["rally_length"].apply(bucket_rally)

        self.df["serve_length"] = self.df["serve_bounce"].apply(bucket_serves)

        # print(df)

        # Group by rally category and winner
        rally_stats = self.df.groupby(["winner", "rally_category"]).size().unstack(fill_value=0)

        # Total points per rally category
        total_per_category = self.df.groupby(["rally_category"]).size()

        # Win rate per player per category
        winrate_by_rally = rally_stats.div(total_per_category, axis=1)

        # print(rally_stats)
        # print(total_per_category)
        # print(winrate_by_rally)

        print(summary_stats)


        return summary_stats
    
    def filter_stats(self,filters):
        print(filters)
        result_df = self.df[self.df['winner'].isin(filters['Winner']) & self.df['server'].isin(filters['Server']) & self.df['serve_length'].isin(filters['Serve Type']) & self.df['rally_category'].isin(filters['Rally Length'])]
        print(result_df)
        result = result_df["winning_bounce"].tolist()
        #???
        print(result)
        return result