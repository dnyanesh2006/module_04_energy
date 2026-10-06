import requests
import json
import time

url = "http://127.0.0.1:8000/api/module4/analyze"
video_path = r"D:\ELECTRICITY & ENERGY WASTAGE DETECTION\module_04_energy\datasets\test_video.mp4"

print(f"Testing API at {url} with video {video_path}")
start_time = time.time()
with open(video_path, 'rb') as f:
    files = {'video': ('test_video.mp4', f, 'video/mp4')}
    try:
        response = requests.post(url, files=files)
        print(f"Response status: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print("Summary:")
            print(json.dumps(data["summary"], indent=2))
            
            if data["event"]:
                print(f"Event: {data['event']['event_type']} (ID: {data['event']['event_id']})")
            else:
                print("Event: None")
                
            print(f"Evidence Image URL: {data['evidence_image']}")
            print(f"Annotated Video URL: {data['annotated_video']}")
            
            print(f"Time taken: {time.time() - start_time:.2f}s")
        else:
            print("Error:")
            print(response.text)
    except Exception as e:
        print(f"Failed to connect or test: {e}")
