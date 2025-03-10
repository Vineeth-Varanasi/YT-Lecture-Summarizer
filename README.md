# YouTube Lecture Summarizer

A Streamlit application that automatically extracts and summarizes YouTube lecture content using the Mistral 7B AI model.

## Overview

This tool helps students, researchers, and lifelong learners quickly extract the key concepts from educational YouTube videos. The application:

1. Extracts the full transcript from any YouTube video
2. Processes and cleans the transcript to remove filler words and sponsorship content
3. Uses Mistral 7B to generate a concise, comprehensive summary focused on educational content
4. Presents the summary with key points highlighted
5. Allows downloading the summary for later reference

## Features

- **Transcript Extraction**: Automatically pulls transcript data from YouTube videos
- **Intelligent Sampling**: For longer lectures, samples content from throughout the video to ensure comprehensive coverage
- **Adaptive Summarization**: Adjusts summary length based on lecture content length and complexity
- **Configurable Output**: Control summary length with a simple slider
- **Key Points Extraction**: Automatically identifies and highlights the most important concepts
- **Error Handling**: Robust error recovery for API outages or transcript issues

## Requirements

- Python 3.7+
- An API key from Hugging Face (for Mistral 7B model access)

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/youtube-lecture-summarizer.git
   cd youtube-lecture-summarizer
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Create a `.env` file in the project root with your Hugging Face API key:
   ```
   HF_API_KEY=your_huggingface_api_key_here
   ```

## Usage

1. Start the application:
   ```bash
   streamlit run app.py
   ```

2. Open your web browser and navigate to the URL displayed in the terminal (typically http://localhost:8501)

3. Paste a YouTube lecture URL in the input field

4. Adjust the summary length using the slider if desired

5. Click "Generate Summary" and wait for the processing to complete

6. View the summary and download it if needed

## How It Works

1. **Transcript Extraction**: Uses the YouTube Transcript API to extract closed captions
2. **Cleaning**: Removes filler words, sponsorship mentions, and other non-educational content
3. **Sampling**: For longer lectures, strategically samples content from throughout the video
4. **AI Processing**: Sends the processed transcript to Mistral 7B with specific instructions
5. **Summarization**: Generates a concise overview focused on educational content
6. **Key Points**: Extracts the most significant sentences as bullet points

## Limitations

- Requires videos to have closed captions/subtitles available
- Quality depends on the accuracy of the original transcripts
- Very technical content may have specialized terminology that impacts summary quality
- API rate limits may apply based on your Hugging Face account

## Troubleshooting

- **API Key Issues**: Ensure your Hugging Face API key is correctly set in the `.env` file
- **Model Loading Delays**: The system automatically retries if the model is still loading
- **Transcript Errors**: Some videos may have restricted or unavailable transcripts
- **Long Processing Times**: Longer videos will take more time to process

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Uses the [Mistral 7B](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2) language model from Mistral AI
- Built with [Streamlit](https://streamlit.io/)
- Transcript extraction powered by [youtube-transcript-api](https://github.com/jdepoix/youtube-transcript-api)
