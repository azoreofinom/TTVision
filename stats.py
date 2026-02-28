import pandas as pd
import cv2


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

def bucket_serves_horizontal(serve_pos):
    if serve_pos is None:
        return "Weird"
    
    if serve_pos[0] < 96 or serve_pos[0] > 674:
        return "Long"
    elif serve_pos[0] < 193 or serve_pos[0] > 578:
        return "Half Long"
    else:
        return "Short"




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

        

        # Group by rally category and winner
        rally_stats = self.df.groupby(["winner", "rally_category"]).size().unstack(fill_value=0)

        # Total points per rally category
        total_per_category = self.df.groupby(["rally_category"]).size()

        # Win rate per player per category
        winrate_by_rally = rally_stats.div(total_per_category, axis=1)


        print(summary_stats)


        return summary_stats
    
    def filter_stats(self,filters):
        result_df = self.df[self.df['winner'].isin(filters['Winner']) & self.df['server'].isin(filters['Server']) & self.df['serve_length'].isin(filters['Serve Type']) & self.df['rally_category'].isin(filters['Rally Length'])]
        result = result_df["winning_bounce"].tolist()
        
        return result