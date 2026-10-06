import os
import time
import json
import uuid
import shutil
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
import yaml
import sys

# Import the existing pipeline
from inference.pipeline import process_video

app = FastAPI(title="NEXORA Module 4 API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount paths
app.mount("/static", StaticFiles(directory="static"), name="static")
app.mount("/results/video", StaticFiles(directory="evaluation/results/realtime_test"), name="results_video")
app.mount("/results/evidence", StaticFiles(directory="evaluation/results/evidence"), name="results_evidence")

@app.post("/api/module4/analyze")
async def analyze_video(video: UploadFile = File(...)):
    if not video.filename.lower().endswith(".mp4"):
        raise HTTPException(status_code=400, detail="Only MP4 video files are supported.")
        
    # Read the file
    content = await video.read()
    if len(content) > 100 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="File exceeds the 100 MB maximum size.")
        
    upload_dir = Path("uploads")
    upload_dir.mkdir(exist_ok=True)
    temp_path = upload_dir / f"{uuid.uuid4()}_{video.filename}"
    
    with open(temp_path, "wb") as f:
        f.write(content)
        
    try:
        start_time = time.time()
        
        # Clear previous outputs (optional, but good for cleanliness)
        out_dir = Path("evaluation/results/realtime_test")
        out_dir.mkdir(parents=True, exist_ok=True)
        evidence_dir = Path("evaluation/results/evidence")
        evidence_dir.mkdir(parents=True, exist_ok=True)
        
        # Call existing pipeline
        # Process video will write to evaluation/results/...
        events = process_video(str(temp_path), camera_id="CAM-TEST", config_path="config/config.yaml")
        
        processing_time = time.time() - start_time
        
        # Read the generated outputs
        detection_output_path = out_dir / "detection_output.json"
        event_output_path = out_dir / "event_output.json"
        
        detections = []
        if detection_output_path.exists():
            with open(detection_output_path, "r") as f:
                detections = json.load(f)
                
        # Parse detections to get summary counts
        person_count = 0
        light_on_count = 0
        light_off_count = 0
        
        for d in detections:
            if d["class_name"] == "person": person_count += 1
            if d["class_name"] == "light_on": light_on_count += 1
            if d["class_name"] == "light_off": light_off_count += 1

        # Check occupancy based on person count
        # In this demo, let's just determine if there were tracks
        person_tracks = len(set(d.get("track_id") for d in detections if d["class_name"] == "person" and d.get("track_id") is not None))

        # Check for event
        event_data = None
        evidence_url = None
        if events and len(events) > 0:
            event_data = events[0]  # Just take the first event for the simple UI
            evidence_file = evidence_dir / f"{event_data['event_id']}.jpg"
            if evidence_file.exists():
                evidence_url = f"/results/evidence/{event_data['event_id']}.jpg"

        video_url = "/results/video/annotated_test.mp4"
        
        # We need the original video duration/resolution, but we can simulate or get it from OpenCV in a real case.
        # We'll return the raw data and let the frontend format it.
        
        return JSONResponse({
            "status": "success",
            "summary": {
                "filename": video.filename,
                "processing_time": processing_time,
                "person_detected": person_count > 0,
                "person_count": person_count,
                "light_on_detected": light_on_count > 0,
                "light_on_count": light_on_count,
                "light_off_detected": light_off_count > 0,
                "light_off_count": light_off_count,
                "person_tracks": person_tracks
            },
            "event": event_data,
            "evidence_image": evidence_url,
            "annotated_video": video_url,
            "raw_output": {
                "events": events,
                "detections": detections[:10]  # Return only first 10 for UI rendering so it doesn't crash browser
            }
        })
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Video analysis failed: {str(e)}")
    finally:
        # Cleanup uploaded file
        if temp_path.exists():
            os.remove(temp_path)

@app.get("/")
def read_root():
    return JSONResponse({"status": "NEXORA Module 4 API is running. Access UI at /static/index.html"})
