import requests
import json

def query_spacetrack():
    try:
        query_url = '' 
        
        username = ""
        password = ""
        
        response = requests.get(query_url, auth=(username, password))
        response.raise_for_status()
        
        filename = "placeholder.json"
        with open(filename, "w") as f:
            f.write(response.text)
            
        print(f"Saved {filename}")
        
    except Exception as e:
        print(f"Error: {e}")
        return None


if __name__ == '__main__':
    query_spacetrack()