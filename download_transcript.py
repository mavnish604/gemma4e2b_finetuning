import os
import re
import yt_dlp

# --- Configuration ---
PLAYLIST_URL = 'https://youtube.com/playlist?list=PLSw9nXpYo3m6Bb_MVBgd0jRE3U4hJvZkd&si=XJ-Bkx4j5429OLhy'
RAW_DIR = 'raw_subtitles'
CLEAN_DIR = 'gemma_dataset_txt'

def download_transcripts():
    """Downloads English transcripts (manual & auto) using yt-dlp."""
    if not os.path.exists(RAW_DIR):
        os.makedirs(RAW_DIR)

    import subprocess
    print(f"🚀 Starting extraction for {PLAYLIST_URL}...")
    
    cmd = [
        'yt-dlp',
        '--skip-download',
        '--write-subs',
        '--write-auto-subs',
        '--sub-langs', 'en.*',
        '--convert-subs', 'srt',
        '-o', f'{RAW_DIR}/%(title)s.%(ext)s',
        '--ignore-errors',
        PLAYLIST_URL
    ]
    
    subprocess.run(cmd)

def clean_srt_to_text():
    """Converts messy SRT files into clean paragraphs for fine-tuning."""
    if not os.path.exists(CLEAN_DIR):
        os.makedirs(CLEAN_DIR)

    print(f"🧹 Cleaning SRTs into plain text for Gemma...")
    
    for filename in os.listdir(RAW_DIR):
        # Focus on English subtitle files
        if filename.endswith('.srt'):
            filepath = os.path.join(RAW_DIR, filename)
            
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()

            # Split into blocks by timestamps
            blocks = re.split(r'\d+\n\d{2}:\d{2}:\d{2},\d{3} --> \d{2}:\d{2}:\d{2},\d{3}', content)
            
            dedup_lines = []
            prev_lines = []
            
            for block in blocks:
                # Strip HTML tags and whitespace
                block_text = re.sub(r'<[^>]*>', '', block).strip()
                if not block_text:
                    continue
                
                # Get non-empty lines in this block
                current_lines = [line.strip() for line in block_text.split('\n') if line.strip()]
                
                # Add lines that weren't in the previous block
                for line in current_lines:
                    if line not in prev_lines:
                        dedup_lines.append(line)
                        
                prev_lines = current_lines

            clean_text = ' '.join(dedup_lines)

            # Save as clean .txt file
            clean_name = filename.replace('.en.srt', '.txt').replace('.en-orig.srt', '.txt')
            with open(os.path.join(CLEAN_DIR, clean_name), 'w', encoding='utf-8') as f_out:
                f_out.write(clean_text)

    print(f"✅ Success! Your training data is ready in: {CLEAN_DIR}/")

if __name__ == "__main__":
    download_transcripts()
    clean_srt_to_text()
