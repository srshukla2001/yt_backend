from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import google.generativeai as genai
from youtube_transcript_api import YouTubeTranscriptApi
import re
import os
from dotenv import load_dotenv
import requests

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

def get_video_details(video_id: str):
    api_key = os.getenv("YOUTUBE_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="YouTube API key not configured")

    api_url = f"https://www.googleapis.com/youtube/v3/videos?part=snippet,contentDetails,statistics&id={video_id}&key={api_key}"

    try:
        response = requests.get(api_url)
        response.raise_for_status()
        data = response.json()

        if not data["items"]:
            raise HTTPException(status_code=404, detail="Video not found")

        video_info = data["items"][0]
        snippet = video_info.get("snippet", {})
        statistics = video_info.get("statistics", {})
        content_details = video_info.get("contentDetails", {})

        return {
            "title": snippet.get("title", ""),
            "description": snippet.get("description", ""),
            "tags": snippet.get("tags", []),
            "thumbnail": snippet.get("thumbnails", {}).get("high", {}).get("url", ""),
            "view_count": int(statistics.get("viewCount", 0)),
            "duration": content_details.get("duration", "")
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch video details: {str(e)}")

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
            "Please analyze all of this and provide SEO recommendations in JSON format with the following structure:\n"
            "{\n"
            "  \"video_summary\": {\n"
            "    \"summary\": \"A detailed summary of the video, breaking down key moments, concepts, and the narrative based on the transcript. The summary should be concise but give a good understanding of what the video covers.\",\n"
            "    \"key_themes\": \"A comprehensive breakdown of the main themes or topics discussed throughout the video, including any subtopics or related discussions that are central to the content.\",\n"
            "    \"actionable_takeaways\": \"The most important actionable points that viewers should take away from the video. This should include any advice, lessons, or insights that are emphasized in the content.\",\n"
            "    \"target_audience\": \"A detailed profile of the intended audience for the video based on the content, tone, style, and language. This should include demographic and psychographic characteristics such as age, interests, and expertise level.\",\n"
            "    \"content_score\": \"A rating from 1 to 10 evaluating the overall quality, depth, and relevance of the video content. Justify your score by explaining why the content is valuable or not to the intended audience.\",\n"
            "    \"engagement_analysis\": \"A thorough assessment of the video’s engagement so far (likes, comments, shares), including whether it's attracting the right audience, and suggestions for improving these metrics. Highlight any patterns or trends in viewer interaction.\"\n"
            "  },\n"
            "  \"title_suggestions\": {\n"
            "    \"optimized_title\": \"An optimized title that includes high-impact keywords, emotional appeal, and curiosity. The title should also be clear and concise, not exceeding 70 characters, and should be enticing enough to drive clicks while accurately reflecting the content.\",\n"
            "    \"title_tips\": \"Detailed tips on improving the title's effectiveness, such as including emotional hooks, addressing viewer pain points, using power words, and ensuring that high-traffic keywords are placed at the beginning. Address potential pitfalls like clickbait and title misalignment.\",\n"
            "    \"title_analysis\": \"A detailed analysis of the current title, including its keyword optimization, readability, and emotional appeal. Identify any issues such as vague wording, lack of clarity, or missed keyword opportunities.\",\n"
            "    \"emotional_appeal\": \"In-depth evaluation of the emotional appeal of the title. Consider how well the title triggers curiosity, urgency, or excitement. Suggest ways to evoke stronger emotions and create a stronger psychological connection with potential viewers.\"\n"
            "  },\n"
            "  \"description_suggestions\": {\n"
            "    \"optimized_description\": \"A detailed, keyword-rich description suggestion that not only summarizes the video’s content but also incorporates high-ranking keywords, provides context, includes timestamps for key moments, and encourages viewers to take specific actions (e.g., subscribing, watching other videos).\",\n"
            "    \"description_tips\": \"Actionable tips for optimizing the description, such as proper keyword placement, the inclusion of related video links, timestamps for easy navigation, and using a CTA (Call to Action) like inviting comments or subscriptions. Provide guidance on balancing between SEO and readability.\",\n"
            "    \"description_analysis\": \"An in-depth review of the current description, assessing the clarity, keyword usage, structure, and whether the description is optimized for both search engines and human readers. Identify opportunities to improve the description’s discoverability and engagement potential.\",\n"
            "    \"description_length\": \"Provide detailed recommendations on the optimal length for descriptions based on current best practices. Consider factors like SEO ranking, user engagement, and whether the description length aligns with YouTube’s algorithms and audience expectations.\"\n"
            "  },\n"
            "  \"tag_suggestions\": {\n"
            "    \"new_tags\": \"Comprehensive suggestions for new tags that could improve discoverability, including long-tail keywords, trending keywords, and niche-specific terms. Include both highly searched and low-competition keywords to target a broad and specific audience.\",\n"
            "    \"tag_analysis\": \"A deep analysis of the current tags and how they could be improved. Discuss the balance between broad and niche tags, and suggest any missed opportunities. Assess whether the current tags are helping or hindering search visibility.\"\n"
            "    \"tag_density\": \"Evaluate the current tag density and suggest whether adding or removing tags could enhance search discoverability. Consider the number of tags used, their relevance to the video’s content, and their impact on SEO.\"\n"
            "  },\n"
            "  \"thumbnail_analysis\": {\n"
            "    \"thumbnail_improvement\": \"Specific, actionable suggestions for improving the thumbnail’s visual elements, such as brightness, contrast, font usage, text legibility, and inclusion of faces or other attention-grabbing elements. Discuss how the thumbnail aligns with the video content and audience preferences.\",\n"
            "    \"best_thumbnail_practices\": \"A comprehensive summary of best practices for creating high-CTR thumbnails, including current design trends, color psychology, font choice, and the importance of text-overlay positioning. Also, evaluate the thumbnail size and format for various devices.\",\n"
            "    \"visual_analysis\": \"A detailed assessment of the existing thumbnail, focusing on factors like color contrast, text size and readability, the use of imagery (e.g., faces or illustrations), and how well it communicates the video’s core message.\"\n"
            "    \"CTR_analysis\": \"A performance-based analysis of the thumbnail’s click-through rate (CTR), and suggestions for improving CTR, including A/B testing thumbnails and using eye-catching visuals. Discuss how thumbnails should change based on content type and viewer demographics.\"\n"
            "  },\n"
            "  \"additional_insights\": {\n"
            "    \"engagement_tips\": \"Advanced strategies for increasing engagement, such as incorporating social proof, asking viewers specific questions, encouraging viewers to share the video, or adding interactive elements like polls and quizzes. Consider methods to boost retention and comments.\",\n"
            "    \"SEO_best_practices\": \"A comprehensive list of SEO best practices for YouTube, including optimizing metadata, using closed captions, maintaining consistent video uploads, promoting videos across platforms, creating playlists, and interacting with the community to increase visibility and ranking.\",\n"
            "    \"actionable_improvements\": \"A prioritized list of actionable improvements across all areas—title, description, tags, and thumbnail—along with estimated SEO and engagement impacts based on research and current YouTube trends.\",\n"
            "    \"social_media_strategies\": \"Detailed strategies for promoting the video on social media platforms such as Twitter, Instagram, and Facebook. Focus on how to align social media campaigns with YouTube SEO efforts and boost cross-platform visibility.\"\n"
            "  },\n"
            "  \"performance_predictions\": {\n"
            "    \"expected_performance\": \"An in-depth analysis predicting the video’s performance (e.g., views, likes, comments, shares) over different timeframes based on the suggested SEO changes. Estimate how SEO optimization will affect metrics like watch time and CTR.\",\n"
            "    \"strategic_actions\": \"Step-by-step strategies to follow if the video’s performance doesn’t meet expectations, such as re-optimizing metadata, cross-promoting, revisiting thumbnail design, or improving content quality.\",\n"
            "    \"estimated_SEO_impact\": \"A precise estimate of how the SEO improvements will affect the video’s ranking, impressions, and views within different time frames (e.g., 24 hours, 1 week, 1 month).\"\n"
            "  },\n"
            "  \"video_trends\": {\n"
            "    \"current_trends\": \"In-depth insights into current YouTube trends, including the latest viral challenges, hot topics, or emerging content formats that the video can align with to increase visibility. Highlight trends within the content’s niche.\",\n"
            "    \"niche_trends\": \"A detailed analysis of trends specific to the video's niche, including content format, language use, viewer preferences, and trending keywords. Explain how the video can ride on these niche trends to stand out.\"\n"
            "  }\n"
            "}\n"
            "Ensure the response is formatted strictly as valid JSON without any additional keywords, formatting, or explanations. "
            "The response must be a plain JSON object as requested."
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
        gemini_api_key = os.getenv("GEMINI_API_KEY")
        if not gemini_api_key:
            raise HTTPException(status_code=500, detail="Gemini API key not configured")

        # Extract video ID and get details
        video_id = extract_video_id(video.url)
        video_info = get_video_details(video_id)
        
        # Get transcript
        transcript_text = get_youtube_transcript(video_id)
        
        # Extract video details
        title = video_info.get("title", "")
        description = video_info.get("description", "")
        tags = video_info.get("tags", [])
        thumbnail_url = video_info.get("thumbnail", "")
        views = video_info.get("view_count", 0)
        length = video_info.get("duration", "")
        
        # Get Gemini analysis
        gemini_analysis = analyze_video_with_gemini(
            transcript_text,
            thumbnail_url,
            title,
            description,
            tags,
            gemini_api_key
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
