from ionop_czml import ionop_czml
from live.build_czml import build_czml

if __name__ == '__main__':

    # now build the czml files
    # try:
        
    #     # df and distance_matrix need to come from the clustering code and get fed into here.
    #     df = ...
    #     distance_matrix, key = ...
                
    #     if df.empty:
    #         print("No data available to plot.")
    #         exit(0)
            
    #     build_czml(df)
        
    # except Exception as e:
    #     print(f"Error: {e}")
    
    ACCESSTOKEN = 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJqdGkiOiI5OTMwYjJlMS0yYjBhLTQwMmMtYjJkZi1mZWZiY2RiYTNmN2UiLCJpZCI6MjQwODIwLCJpYXQiOjE3MzgzMDM2ODl9.h1pXOgujWRPoS6ZFc5wL-l5_XJnSyUsPZym3ssZj7TQ'
  
    try:
        ionop_czml(ACCESSTOKEN)

    except Exception as e:
        print(f"Error: {e}")
        
    
    