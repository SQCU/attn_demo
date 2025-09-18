# batch_encode_for_twitter.py
import os
import subprocess
import glob

# --- CONFIG ---
AUDIO_DIR = "generated_audio/curated"
COVER_ART = "generated_audio/nothing-ever-markovs.png" # Make sure this file exists
OUTPUT_DIR = "twitter_videos"
# ---

os.makedirs(OUTPUT_DIR, exist_ok=True)

if not os.path.exists(COVER_ART):
    print(f"Error: Cover art '{COVER_ART}' not found. Please create this image file.")
else:
    wav_files = glob.glob(os.path.join(AUDIO_DIR, "*.wav"))
    
    for wav_file in wav_files:
        basename = os.path.splitext(os.path.basename(wav_file))[0]
        output_file = os.path.join(OUTPUT_DIR, f"{basename}.mp4")
        
        print(f"Encoding {wav_file}...")
        
        command = [
            'ffmpeg',
            '-loop', '1',
            '-framerate', '2',
            '-i', COVER_ART,
            '-i', wav_file,
            '-c:v', 'libx264',
            '-preset', 'medium',
            '-tune', 'stillimage',
            '-crf', '18',
            '-c:a', 'aac',
            '-b:a', '192k',
            '-pix_fmt', 'yuv420p',
            '-shortest',
            output_file
        ]
        
        # Use -y to automatically overwrite existing files
        command.insert(1, '-y')
        
        try:
            subprocess.run(command, check=True, capture_output=True, text=True)
            print(f"Successfully created {output_file}")
        except subprocess.CalledProcessError as e:
            print(f"--- FFMPEG FAILED for {wav_file} ---")
            print(f"STDERR:\n{e.stderr}")