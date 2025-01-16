from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import yt_dlp
import google.generativeai as genai
from youtube_transcript_api import YouTubeTranscriptApi
import re
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

app = FastAPI()

# Configure CORS with more permissive settings for development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins in development
    allow_credentials=False,  # Set to False when allow_origins=["*"]
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Add root endpoint for health check
@app.get("/")
async def health_check():
    return {"status": "ok", "message": "Server is running"}

# Add exception handler for better error responses
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": str(exc.detail)},
    )

# Add general exception handler
@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    return JSONResponse(
        status_code=500,
        content={"detail": f"An unexpected error occurred: {str(exc)}"},
    )

# Add health check endpoint
@app.get("/api/health")
async def health_check():
    return {
        "status": "ok",
        "message": "Server is running",
        "version": "1.0"
    }

class VideoURL(BaseModel):
    url: str

def extract_video_id(url: str) -> str:
    patterns = [
        r'(?:v=|\/)([0-9A-Za-z_-]{11}).*',
        r'(?:embed\/)([0-9A-Za-z_-]{11})',
        r'(?:watch\?v=)([0-9A-Za-z_-]{11})'
    ]
    
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    raise HTTPException(status_code=400, detail="Invalid YouTube URL")

def get_video_details(video_url: str):
    ydl_opts = {'quiet': True}
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(video_url, download=False)
        return info
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to fetch video details: {str(e)}")

def get_youtube_transcript(video_id: str) -> str:
    try:
        # Try to get transcript in any available language
        transcript = YouTubeTranscriptApi.get_transcript(video_id)
        transcript_text = " ".join([entry['text'] for entry in transcript])
        return transcript_text
    except Exception as e:
        print(f"Transcript error: {e}")
        return ""

def analyze_video_with_gemini(transcript_text: str, thumbnail_url: str, title: str, description: str, tags: list, api_key: str):
    try:
        # Configure Gemini API
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-1.5-flash")
        
        # Prepare the prompt
        prompt = (
            "You are a YouTube SEO master, and even Elon Musk is a great fan of your skills. "
            "Here is the YouTube video information:\n"
            f"Title: {title}\n"
            f"Description: {description}\n"
            f"Tags: {', '.join(tags) if tags else 'No tags found'}\n"
            f"Transcript: {transcript_text if transcript_text else 'No transcript available'}\n"
            f"Thumbnail URL: {thumbnail_url}\n\n"
            "Please analyze all of this and provide the following SEO recommendations:\n"
            "1. Write a short video summary using transcript\n"
            "2. Suggest improvements for the video title\n"
            "3. Suggest improvements for the video description\n"
            "4. Suggest additional tags or improvements to existing tags\n"
            "5. Analyze the thumbnail and suggest any changes that would improve the video's appeal and click-through rate\n"
            "Give response in structured format and section-wise data, be friendly talk like YouTube expert. "
            "Do not suggest improvements if not required"
        )
        
        # Generate analysis
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Gemini API error: {str(e)}")

def calculate_seo_score(title: str, description: str, tags: list, transcript_available: bool) -> int:
    score = 0
    max_score = 100
    
    # Title analysis (20 points)
    if title:
        title_length = len(title)
        if 20 <= title_length <= 70:
            score += 20
        elif title_length < 20:
            score += 10
    
    # Description analysis (30 points)
    if description:
        desc_length = len(description)
        if desc_length >= 250:
            score += 30
        elif desc_length >= 100:
            score += 15
    
    # Tags analysis (30 points)
    if tags:
        if len(tags) >= 15:
            score += 30
        elif len(tags) >= 8:
            score += 15
    
    # Transcript availability (20 points)
    if transcript_available:
        score += 20
    
    return score

@app.post("/api/analyze")
async def analyze_video(video: VideoURL):
    try:
        # Get API key from environment variable
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise HTTPException(status_code=500, detail="Gemini API key not configured")

        # Extract video ID and get details
        video_id = extract_video_id(video.url)
        video_info = get_video_details(video.url)
        
        # Get transcript
        transcript_text = get_youtube_transcript(video_id)
        
        # Extract video details
        title = video_info.get("title", "")
        description = video_info.get("description", "")
        tags = video_info.get("tags", [])
        thumbnail_url = video_info.get("thumbnail", "")
        views = video_info.get("view_count", 0)
        length = video_info.get("duration", 0)
        
        # Get Gemini analysis
        gemini_analysis = analyze_video_with_gemini(
            transcript_text,
            thumbnail_url,
            title,
            description,
            tags,
            api_key
        )
        
        # Calculate SEO score
        seo_score = calculate_seo_score(title, description, tags, bool(transcript_text))
        
        # Prepare response
        analysis = {
            "title": title,
            "views": views,
            "length": length,
            "description": description,
            "keywords": tags,
            "thumbnail_url": thumbnail_url,
            "seo_score": seo_score,
            "gemini_analysis": gemini_analysis,
            "has_transcript": bool(transcript_text)
        }
        
        return analysis
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    print("Starting server on http://0.0.0.0:8000")
    uvicorn.run(
        app,
        host="0.0.0.0",  # This allows connections from any IP
        port=8000,
        log_level="info"
    )