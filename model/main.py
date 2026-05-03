from scoring import scoring_main
from ionop_czml import ionop_czml
from live.build_czml import build_czml_live
import pandas as pd

if __name__ == '__main__':
    # builds the czml files
    try:
        print("Scoring started")
        scoring_main()
        
    except Exception as e:
        print(f"Error: {e}")
        
    print("Scoring done")
    
    # now build the czml files
    try:
        results_df = pd.read_pickle("data/satellites_with_scores.pkl")
        build_czml_live(results_df)
    except Exception as e:
        print(f"Error: {e}")
    
    ACCESSTOKEN_live = 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJqdGkiOiJkYjlmZDMxMy1lY2RmLTQyMDMtYTZhZS0wMmY4MzcyZDc4ZGEiLCJpZCI6MjQwODIwLCJpYXQiOjE3NzcxOTU0NTB9.M5SBzZZAebNNCMhZbSUqozkEm8LIul3abqANtS8URrU'
    try:
        ionop_czml(ACCESSTOKEN_live, 'live')
    except Exception as e:
        print(f"Error: {e}")
        
    
    