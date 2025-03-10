import streamlit as st
from dotenv import load_dotenv
import os
from pathlib import Path
import requests
import time
from youtube_transcript_api import YouTubeTranscriptApi

load_dotenv()
HF_API_KEY = os.getenv("HF_API_KEY")
HF_API_URL = "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.2"

def extract_video_id(youtube_url):
    """Extract YouTube video ID from various URL formats"""
    if "youtube.com/watch?v=" in youtube_url:
        video_id = youtube_url.split("watch?v=")[1]
        if "&" in video_id:
            video_id = video_id.split("&")[0]
        return video_id
    elif "youtu.be/" in youtube_url:
        return youtube_url.split("youtu.be/")[1]
    elif "youtube.com/embed/" in youtube_url:
        return youtube_url.split("embed/")[1]
    else:
        raise ValueError("Not a valid YouTube URL")

def get_transcript(youtube_video_url):
    try:
        video_id = extract_video_id(youtube_video_url)
        transcript_text = YouTubeTranscriptApi.get_transcript(video_id)
        transcript = ""
        for i in transcript_text:
            transcript += " " + i["text"]
        return transcript
    except Exception as e:
        st.error(f"Failed to retrieve transcript: {str(e)}")
        raise e

def clean_transcript(transcript):
    """Clean the transcript to make it more summarizable"""
    filler_words = ["um", "uh", "like", "you know", "sort of", "kind of"]
    cleaned = transcript
    for word in filler_words:
        cleaned = cleaned.replace(f" {word} ", " ")
    
    import re
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    
    sponsorship_patterns = [
        r"sponsor[a-z]* (this|today)['s]* (video|episode)",
        r"check out [^.]*in the description",
        r"don't forget to (like|subscribe)",
        r"hit the (like button|notification bell)"
    ]
    
    for pattern in sponsorship_patterns:
        cleaned = re.sub(pattern, "", cleaned, flags=re.IGNORECASE)
    
    words = cleaned.split()
    if len(words) > 200:  
        ten_percent = len(words) // 10
        cleaned = " ".join(words[ten_percent:-ten_percent])
    
    return cleaned

def sample_transcript(transcript, max_samples=5, sample_size=200):
    """
    Sample content from throughout a long transcript to get representative segments
    
    Args:
        transcript (str): The full transcript text
        max_samples (int): Number of samples to take
        sample_size (int): Approximate words per sample
        
    Returns:
        str: A concatenated string of samples with markers
    """
    words = transcript.split()
    total_words = len(words)
    
    
    if total_words <= max_samples * sample_size:
        return transcript
    
    
    intervals = total_words // max_samples
    
    
    samples = []
    for i in range(max_samples):
        start_idx = min(i * intervals, total_words - sample_size)
        end_idx = min(start_idx + sample_size, total_words)
        
        
        sample = " ".join(words[start_idx:end_idx])
        
        
        if i > 0:
            sample = f"\n[...content continues...]\n\n{sample}"
            
        samples.append(sample)
    
    
    samples.append("\n[...lecture concludes...]")
    
    # Join all samples
    return " ".join(samples)

def calculate_token_length(text):
    """Estimate token count based on words (rough approximation)"""
    return len(text.split()) * 1.3  # Rough estimate: 1 word ≈ 1.3 tokens

def generate_reply(transcript_text):
    if not HF_API_KEY:
        st.error("HF_API_KEY not found. Please check your .env file.")
        return "Error: API key not configured"
        
    headers = {"Authorization": f"Bearer {HF_API_KEY}"}
    
    # Clean the transcript
    cleaned_transcript = clean_transcript(transcript_text)
    
   
    if len(cleaned_transcript.split()) > 1000:  
        sampled_transcript = sample_transcript(cleaned_transcript)
        st.info("This is a long lecture. Creating a general overview of the entire content.")
    else:
        sampled_transcript = cleaned_transcript
    
    
    max_tokens_available = 4000
    transcript_tokens = calculate_token_length(sampled_transcript)
    

    max_summary_tokens = min(495, max_tokens_available - int(transcript_tokens) - 100)
    max_summary_words = int(max_summary_tokens / 1.3)  
    
    
    if len(sampled_transcript.split()) < 500:
        target_length = min(max_summary_words, 150)  
    elif len(sampled_transcript.split()) < 2000:
        target_length = min(max_summary_words, 250)  
    else:
        target_length = min(max_summary_words, 400)  
    
    
    instruction = f"""You are an educational content summarizer. 
    Provide a concise yet comprehensive overview of this lecture.
    Focus only on the key educational concepts and main points.
    This should be a general overview that covers ALL main topics in the lecture.
    Even if you don't have all details, include mentions of ALL topic areas covered.
    Ignore any introductions, like/subscribe reminders, or sponsorship mentions.
    Your summary must be complete and properly concluded - no unfinished sentences.
    TARGET LENGTH: {target_length} words exactly. This is important - use all available space but do not exceed this limit."""
    
    prompt = f"<s>[INST] {instruction}\n\n{sampled_transcript} [/INST]"
    
    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": max_summary_tokens,  
            "temperature": 0.6,     
            "top_p": 0.9,
            "do_sample": True,
            "return_full_text": False  
        }
    }
    
    max_retries = 3
    for attempt in range(max_retries):
        try:
            with st.spinner(f"Generating summary (attempt {attempt+1}/{max_retries})..."):
                response = requests.post(HF_API_URL, headers=headers, json=payload, timeout=120)
                
                if response.status_code == 200:
                    result = response.json()
                    
                    
                    if isinstance(result, list) and len(result) > 0:
                        generated_text = result[0].get("generated_text", "")
                        if not generated_text:
                            generated_text = str(result[0])
                    else:
                        generated_text = result.get("generated_text", "")
                        if not generated_text:
                            generated_text = str(result)
                    
                    
                    if "[/INST]" in generated_text:
                        summary = generated_text.split("[/INST]")[1].strip()
                    else:
                        summary = generated_text.strip()
                    
                    
                    current_word_count = len(summary.split())
                    st.sidebar.info(f"Generated summary: {current_word_count} words ({int(calculate_token_length(summary))} est. tokens)")
                    
                    
                    if len(summary) < 50:  
                        raise ValueError(f"Received too short summary: {summary}")
                    
                    
                    if not summary.endswith(('.', '!', '?', '"', ')', ']')):
                        summary += "..." 
                        st.warning("The summary may be incomplete due to length limitations.")
                        
                    return summary
                
                elif response.status_code == 503:
                    wait_time = (2 ** attempt) * 1.5
                    st.sidebar.warning(f"Model is loading, retrying in {wait_time:.1f} seconds... (Attempt {attempt+1}/{max_retries})")
                    time.sleep(wait_time)
                else:
                    error_msg = f"API Error: {response.status_code}"
                    try:
                        error_details = response.json()
                        error_msg += f" - {str(error_details)}"
                    except:
                        pass
                    st.sidebar.warning(f"{error_msg}, trying again...")
                    time.sleep(2)
                    
        except requests.exceptions.Timeout:
            st.sidebar.warning(f"Request timed out, retrying (attempt {attempt+2}/{max_retries})...")
            time.sleep(2)
        except Exception as e:
            st.sidebar.warning(f"Request failed: {str(e)}, retrying...")
            time.sleep(2)
    
    return "Unable to generate summary. The API service may be temporarily unavailable. Please try again later."

def format_summary(summary):
    import re
    
    
    if not summary:
        return "### Error\n\nFailed to generate a summary."
    
    
    formatted = "### Summary\n\n"
    formatted += summary
    
    
    if "•" not in summary and "-" not in summary:
        formatted += "\n\n### Key Points\n"
        sentences = re.split(r'(?<=[.!?])\s+', summary)
        
        
        content_sentences = []
        for s in sentences:
            
            if re.search(r"(subscribe|like|comment|sponsor|watch|video|channel)", s, re.IGNORECASE):
                continue
            if len(s.split()) > 5:
                content_sentences.append(s)
        
        
        for sentence in content_sentences[:5]:
            formatted += f"\n• {sentence}"
    
    return formatted

st.set_page_config(layout="wide", page_title="YouTube Lecture Summarizer")


st.title("YouTube Lecture Summarizer")
st.markdown("---")


with st.sidebar:
    st.header("Input Options")
    youtube_link = st.text_input("Enter YouTube lecture link:")
    
    
    st.subheader("Summary Options")
    summary_length = st.slider(
        "Summary Length",
        min_value=150,
        max_value=500,
        value=300,
        step=50,
        help="Approximate target word count for the summary"
    )
    
    generate_button = st.button("Generate Summary", use_container_width=True)
    


if youtube_link:
    try:
        video_id = extract_video_id(youtube_link)
        
        st.image(f"http://img.youtube.com/vi/{video_id}/0.jpg", use_container_width=True)
        
        
        transcript_placeholder = st.empty()
        summary_placeholder = st.empty()
        
        
        if generate_button:
            try:
                with st.spinner("Extracting transcript..."):
                    transcript = get_transcript(youtube_link)
                    
                    if transcript:
                        
                        word_count = len(transcript.split())
                        st.info(f"Extracted transcript with {word_count} words")
                        
                        
                        with st.spinner("Generating summary..."):
                            summary = generate_reply(transcript)
                        
                        with summary_placeholder.container():
                            st.markdown("### Lecture Summary")
                            st.markdown(format_summary(summary))
                            
                            
                            col1 = st.columns(1)
                            with col1[0]:
                                if summary:
                                    st.download_button(
                                        label="Download Summary",
                                        data=summary,
                                        file_name="lecture_summary.txt",
                                        mime="text/plain"
                                    )
            except Exception as e:
                import traceback
                error_details = traceback.format_exc()
                st.error(f"An error occurred: {str(e)}\n\nDetails: {error_details[:500]}...")
    except Exception as e:
        st.error(f"Invalid YouTube URL: {str(e)}")