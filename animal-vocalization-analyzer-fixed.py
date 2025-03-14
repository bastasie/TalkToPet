# Animal Vocalization Analyzer
# For use in Google Colab

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import librosa
import librosa.display
import IPython.display as ipd
from scipy import signal
from scipy.io import wavfile  # Use scipy.io.wavfile instead of librosa.output
import requests
import io
import os
from urllib.parse import urlparse
import urllib.request
import tempfile
import warnings
warnings.filterwarnings('ignore')

# Display markdown
from IPython.display import Markdown, display
def printmd(string):
    display(Markdown(string))

printmd("# 🔊 Animal Vocalization Analyzer")
printmd("### Analyzing and decoding animal communications")

# Since the provided links are to Storyblocks catalog pages and not direct audio files,
# we'll provide instructions for the user to download them first.
printmd("""
## Setup Instructions

This notebook can analyze animal vocalizations from:
1. Direct links to audio files (.mp3, .wav)
2. Uploaded audio files

For best results, use high-quality recordings with minimal background noise.
""")

# Function to download audio from direct URLs
def download_audio(url, save_as=None):
    """
    Download audio from a direct URL
    Returns path to the downloaded file
    """
    if save_as is None:
        parsed_url = urlparse(url)
        filename = os.path.basename(parsed_url.path)
        save_as = filename
    
    try:
        # For direct audio file URLs
        if url.endswith('.mp3') or url.endswith('.wav'):
            urllib.request.urlretrieve(url, save_as)
            print(f"Downloaded to {save_as}")
            return save_as
        
        # Try to handle Storyblocks URLs by finding audio source
        else:
            response = requests.get(url)
            if response.status_code == 200:
                # This is a very simplified approach and might not work for all cases
                # Ideally, you would parse the HTML and find the audio source
                print("URL is not a direct audio file. Please download manually.")
                return None
            else:
                print(f"Failed to access URL: {response.status_code}")
                return None
    except Exception as e:
        print(f"Error downloading audio: {e}")
        return None

# Sample direct URLs (these would be provided by the user)
audio_examples = {
    "Example 1: Wolf Howl": "https://freesound.org/data/previews/398/398713_5121236-lq.mp3",
    "Example 2: Dolphin": "https://cdn.freesound.org/previews/257/257855_4843504-lq.mp3",
    "Example 3: Chimp Calls": "https://cdn.freesound.org/previews/338/338428_5865517-lq.mp3",
    "Example 4: Bird Songs": "https://cdn.freesound.org/previews/561/561320_2160092-lq.mp3"
}

printmd("## Sample Audio URLs")
printmd("These are examples of direct audio URLs that can be analyzed:")
for name, url in audio_examples.items():
    printmd(f"* **{name}**: [Link]({url})")

# Function to explore and analyze an audio file
def analyze_audio(file_path=None, url=None, species_hint=None):
    """
    Analyze animal vocalizations from an audio file or URL
    """
    # Load the audio from file or URL
    if file_path:
        try:
            y, sr = librosa.load(file_path, sr=None)
            source = file_path
        except Exception as e:
            print(f"Error loading audio file: {e}")
            return
    elif url:
        try:
            response = requests.get(url)
            if response.status_code == 200:
                y, sr = librosa.load(io.BytesIO(response.content), sr=None)
                source = url
            else:
                print(f"Failed to download from URL: {response.status_code}")
                return
        except Exception as e:
            print(f"Error loading audio from URL: {e}")
            return
    else:
        print("Please provide either a file path or URL")
        return
    
    # Display basic audio information
    duration = librosa.get_duration(y=y, sr=sr)
    printmd(f"## Audio Analysis Results")
    printmd(f"**Source**: {source}")
    printmd(f"**Duration**: {duration:.2f} seconds")
    printmd(f"**Sample Rate**: {sr} Hz")
    
    # Allow playback of the audio
    printmd("### Audio Playback")
    # Changed approach: use IPython Audio directly without attempting to write WAV
    display(ipd.Audio(data=y, rate=sr))
    
    # Waveform visualization
    plt.figure(figsize=(14, 5))
    librosa.display.waveshow(y, sr=sr)
    plt.title('Waveform')
    plt.tight_layout()
    plt.show()
    
    # Spectrogram analysis
    plt.figure(figsize=(14, 8))
    D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
    librosa.display.specshow(D, y_axis='log', x_axis='time', sr=sr)
    plt.title('Spectrogram')
    plt.colorbar(format='%+2.0f dB')
    plt.tight_layout()
    plt.show()
    
    # Mel spectrogram - often better for analyzing animal vocalizations
    plt.figure(figsize=(14, 8))
    M = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=sr/2)
    M_db = librosa.power_to_db(M, ref=np.max)
    librosa.display.specshow(M_db, y_axis='mel', x_axis='time', sr=sr)
    plt.title('Mel Spectrogram')
    plt.colorbar(format='%+2.0f dB')
    plt.tight_layout()
    plt.show()
    
    # Attempt to extract features relevant to animal vocalizations
    # These features can help with species identification and vocalization analysis
    
    # Fundamental frequency estimation (pitch tracking)
    pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
    
    # Get the pitch with highest magnitude per frame
    pitch_values = []
    for i in range(magnitudes.shape[1]):
        index = magnitudes[:,i].argmax()
        pitch_values.append(pitches[index,i])
    
    # Filter out unreasonable values
    pitch_values = np.array([p for p in pitch_values if 20 < p < 3000])
    
    # Display pitch information
    if len(pitch_values) > 0:
        plt.figure(figsize=(14, 4))
        plt.plot(pitch_values)
        plt.title('Fundamental Frequency')
        plt.ylabel('Frequency (Hz)')
        plt.xlabel('Frame')
        plt.tight_layout()
        plt.show()
        
        printmd(f"**Average Pitch**: {np.mean(pitch_values):.2f} Hz")
        printmd(f"**Pitch Range**: {np.min(pitch_values):.2f} - {np.max(pitch_values):.2f} Hz")
    
    # Attempt species identification based on acoustic features
    printmd("### Species Identification")
    
    # Calculate key features that can help identify the species
    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
    spectral_contrast = np.mean(librosa.feature.spectral_contrast(y=y, sr=sr), axis=1)
    zero_crossing_rate = librosa.feature.zero_crossing_rate(y)[0]
    
    # Display feature information
    feature_df = pd.DataFrame({
        'Feature': ['Spectral Centroid (mean)', 'Spectral Bandwidth (mean)', 
                   'Zero Crossing Rate (mean)'],
        'Value': [f"{np.mean(spectral_centroid):.2f} Hz", 
                 f"{np.mean(spectral_bandwidth):.2f} Hz",
                 f"{np.mean(zero_crossing_rate):.4f}"]
    })
    display(feature_df)
    
    # Simple logic for species identification based on acoustic features
    # This is a very simplified approach - real systems would use ML models
    species_profiles = {
        'wolf': {
            'pitch_range': (50, 1500),
            'spectral_centroid': (500, 2000),
            'description': 'Sustained howls with harmonic structure, often with frequency modulation',
            'communication_types': ['Howls', 'Barks', 'Whines', 'Growls'],
            'typical_contexts': ['Territorial marking', 'Pack coordination', 'Social bonding', 'Threat displays']
        },
        'dolphin': {
            'pitch_range': (800, 20000),
            'spectral_centroid': (5000, 15000),
            'description': 'High-pitched clicks, whistles, and burst-pulse sounds with rapid frequency modulation',
            'communication_types': ['Signature whistles', 'Echolocation clicks', 'Burst-pulse sounds', 'Synchronized calls'],
            'typical_contexts': ['Individual identification', 'Navigation', 'Food finding', 'Social coordination']
        },
        'chimpanzee': {
            'pitch_range': (100, 3000),
            'spectral_centroid': (300, 2500),
            'description': 'Complex vocalizations including pant-hoots, barks, and screams with varied pitch',
            'communication_types': ['Pant-hoots', 'Food grunts', 'Alarm calls', 'Soft barks', 'Screams'],
            'typical_contexts': ['Group coordination', 'Food discovery', 'Alarm signals', 'Social bonding', 'Aggression']
        },
        'bird': {
            'pitch_range': (1000, 8000),
            'spectral_centroid': (2000, 6000),
            'description': 'Complex songs with rapid frequency modulation, often with repeating patterns',
            'communication_types': ['Songs', 'Calls', 'Alarms', 'Contact calls'],
            'typical_contexts': ['Territorial defense', 'Mate attraction', 'Flock coordination', 'Danger alerts']
        }
    }
    
    # Helper function to determine confidence
    def calculate_confidence(value, expected_range):
        """Calculate confidence based on how well a value fits in expected range"""
        min_val, max_val = expected_range
        range_width = max_val - min_val
        
        # If value is in range, high confidence
        if min_val <= value <= max_val:
            # Higher confidence if closer to middle of range
            distance_from_center = abs(value - (min_val + range_width/2))
            return min(95, 100 - (distance_from_center / (range_width/2)) * 40)
        
        # If value is outside range, calculate how far outside
        if value < min_val:
            distance = min_val - value
        else:
            distance = value - max_val
        
        # Reduce confidence based on distance outside range
        confidence = max(10, 80 - (distance / range_width) * 100)
        return confidence
    
    # Identify species based on features
    mean_pitch = np.mean(pitch_values) if len(pitch_values) > 0 else 0
    mean_centroid = np.mean(spectral_centroid)
    
    if species_hint:
        # If user specified a species hint, use that but verify
        if species_hint.lower() in species_profiles:
            identified_species = species_hint.lower()
            pitch_confidence = calculate_confidence(mean_pitch, species_profiles[identified_species]['pitch_range'])
            centroid_confidence = calculate_confidence(mean_centroid, species_profiles[identified_species]['spectral_centroid'])
            confidence = (pitch_confidence + centroid_confidence) / 2
        else:
            identified_species = 'unknown'
            confidence = 0
    else:
        # Try to identify based on acoustic features
        confidence_scores = {}
        for species, profile in species_profiles.items():
            pitch_confidence = calculate_confidence(mean_pitch, profile['pitch_range'])
            centroid_confidence = calculate_confidence(mean_centroid, profile['spectral_centroid'])
            confidence_scores[species] = (pitch_confidence + centroid_confidence) / 2
        
        identified_species = max(confidence_scores, key=confidence_scores.get)
        confidence = confidence_scores[identified_species]
    
    printmd(f"**Identified Species**: {identified_species.capitalize()} (Confidence: {confidence:.1f}%)")
    
    if identified_species in species_profiles:
        profile = species_profiles[identified_species]
        printmd(f"**Description**: {profile['description']}")
        
        printmd("**Typical Communication Types**:")
        for comm_type in profile['communication_types']:
            printmd(f"- {comm_type}")
            
        printmd("**Typical Communication Contexts**:")
        for context in profile['typical_contexts']:
            printmd(f"- {context}")
    
    # Extract potential communication segments
    # This uses energy and spectral contrast to find vocalization events
    printmd("### Communication Event Detection")
    
    # Calculate energy
    energy = librosa.feature.rms(y=y)[0]
    
    # Find potential communication segments based on energy thresholds
    threshold = np.mean(energy) + 0.5 * np.std(energy)
    
    # Create frames where energy is above threshold
    mask = energy > threshold
    
    # Find segment boundaries
    boundaries = []
    in_segment = False
    segment_start = 0
    
    for i, val in enumerate(mask):
        if val and not in_segment:
            # Start of segment
            in_segment = True
            segment_start = i
        elif not val and in_segment:
            # End of segment
            in_segment = False
            if i - segment_start > 5:  # Minimum length check (5 frames)
                boundaries.append((segment_start, i))
    
    # If still in segment at the end
    if in_segment and len(mask) - segment_start > 5:
        boundaries.append((segment_start, len(mask)))
    
    # Convert frame numbers to time
    hop_length = 512  # Default in librosa
    boundaries_time = [(b[0] * hop_length / sr, b[1] * hop_length / sr) for b in boundaries]
    
    # Display detected events
    if len(boundaries_time) > 0:
        printmd(f"Detected {len(boundaries_time)} communication events:")
        
        # Draw waveform with highlighted segments
        plt.figure(figsize=(14, 5))
        librosa.display.waveshow(y, sr=sr, alpha=0.5)
        
        for i, (start, end) in enumerate(boundaries_time):
            plt.axvspan(start, end, color='red', alpha=0.3)
            plt.text(start, 0.9, f"{i+1}", fontsize=10, verticalalignment='top')
        
        plt.title('Detected Communication Events')
        plt.tight_layout()
        plt.show()
        
        # Display events table
        events_df = pd.DataFrame({
            'Event #': range(1, len(boundaries_time) + 1),
            'Start Time (s)': [f"{start:.2f}" for start, _ in boundaries_time],
            'End Time (s)': [f"{end:.2f}" for _, end in boundaries_time],
            'Duration (s)': [f"{end-start:.2f}" for start, end in boundaries_time]
        })
        display(events_df)
        
        # Analyze each detected event
        printmd("### Communication Event Analysis")
        
        for i, (start_time, end_time) in enumerate(boundaries_time):
            printmd(f"#### Event {i+1}: {start_time:.2f}s - {end_time:.2f}s")
            
            # Extract segment
            start_sample = int(start_time * sr)
            end_sample = int(end_time * sr)
            segment = y[start_sample:end_sample]
            
            # Allow playback of the segment
            display(ipd.Audio(data=segment, rate=sr))
            
            # Calculate segment-specific features
            if len(segment) > 0:
                segment_pitches, segment_magnitudes = librosa.piptrack(y=segment, sr=sr)
                segment_pitches_values = []
                for j in range(segment_magnitudes.shape[1]):
                    if segment_magnitudes.shape[0] > 0 and segment_magnitudes.shape[1] > 0:
                        index = segment_magnitudes[:,j].argmax()
                        segment_pitches_values.append(segment_pitches[index,j])
                
                # Filter out unreasonable values
                segment_pitches_values = np.array([p for p in segment_pitches_values if 20 < p < 3000])
                
                # Calculate additional features
                segment_spectral_centroid = librosa.feature.spectral_centroid(y=segment, sr=sr)[0]
                segment_spectral_bandwidth = librosa.feature.spectral_bandwidth(y=segment, sr=sr)[0]
                segment_zero_crossing_rate = librosa.feature.zero_crossing_rate(segment)[0]
                
                # Show segment mel spectrogram
                plt.figure(figsize=(10, 4))
                M_segment = librosa.feature.melspectrogram(y=segment, sr=sr, n_mels=128, fmax=sr/2)
                M_db_segment = librosa.power_to_db(M_segment, ref=np.max)
                librosa.display.specshow(M_db_segment, y_axis='mel', x_axis='time', sr=sr)
                plt.title(f'Event {i+1} Mel Spectrogram')
                plt.colorbar(format='%+2.0f dB')
                plt.tight_layout()
                plt.show()
                
                # Calculate temporal changes (useful for identifying modulation patterns)
                pitch_changes = np.diff(segment_pitches_values) if len(segment_pitches_values) > 1 else []
                has_rapid_modulation = len(pitch_changes) > 0 and np.std(pitch_changes) > 50
                has_pitch_trends = len(segment_pitches_values) > 0 and np.ptp(segment_pitches_values) > 500
                
                # Generate interpretation based on species and acoustic features
                interpretation = {}
                if identified_species == 'wolf':
                    if len(segment_pitches_values) > 0 and np.mean(segment_pitches_values) < 300:
                        interpretation['type'] = 'Low Howl'
                        interpretation['meaning'] = 'Long-distance communication or territorial signal'
                    elif has_rapid_modulation:
                        interpretation['type'] = 'Bark or Growl'
                        interpretation['meaning'] = 'Warning or threat display'
                    elif len(segment_pitches_values) > 0 and np.mean(segment_pitches_values) > 800:
                        interpretation['type'] = 'Whine or Yelp'
                        interpretation['meaning'] = 'Submission or distress call'
                    else:
                        interpretation['type'] = 'Wolf Vocalization'
                        interpretation['meaning'] = 'Communication or social call'
                
                elif identified_species == 'dolphin':
                    if has_rapid_modulation and np.mean(segment_spectral_centroid) > 10000:
                        interpretation['type'] = 'Echolocation Click Train'
                        interpretation['meaning'] = 'Object investigation or navigation'
                    elif has_pitch_trends:
                        interpretation['type'] = 'Signature Whistle'
                        interpretation['meaning'] = 'Individual identification or location broadcast'
                    elif len(segment_pitches_values) > 0 and np.mean(segment_pitches_values) > 5000:
                        interpretation['type'] = 'Burst-Pulse Sound'
                        interpretation['meaning'] = 'Social interaction or emotional state expression'
                    else:
                        interpretation['type'] = 'Dolphin Vocalization'
                        interpretation['meaning'] = 'Communication or social call'
                
                elif identified_species == 'chimpanzee':
                    if has_rapid_modulation and np.mean(segment_zero_crossing_rate) > 0.1:
                        interpretation['type'] = 'Scream or Alarm Call'
                        interpretation['meaning'] = 'Distress or warning to group'
                    elif has_pitch_trends and len(segment_pitches_values) > 0 and np.mean(segment_pitches_values) > 1000:
                        interpretation['type'] = 'Pant-hoot'
                        interpretation['meaning'] = 'Long-distance communication or group coordination'
                    elif len(segment_pitches_values) > 0 and np.mean(segment_pitches_values) < 500:
                        interpretation['type'] = 'Food Grunt'
                        interpretation['meaning'] = 'Food discovery or satisfaction'
                    else:
                        interpretation['type'] = 'Chimpanzee Vocalization'
                        interpretation['meaning'] = 'Social or emotional expression'
                
                elif identified_species == 'bird':
                    if has_rapid_modulation and has_pitch_trends:
                        interpretation['type'] = 'Complex Song'
                        interpretation['meaning'] = 'Territory defense or mate attraction'
                    elif len(segment_pitches_values) > 0 and np.std(segment_pitches_values) < 200:
                        interpretation['type'] = 'Contact Call'
                        interpretation['meaning'] = 'Maintaining flock cohesion'
                    elif len(segment_pitches_values) > 0 and np.mean(segment_pitches_values) > 3000:
                        interpretation['type'] = 'Alarm Call'
                        interpretation['meaning'] = 'Warning about potential danger'
                    else:
                        interpretation['type'] = 'Bird Vocalization'
                        interpretation['meaning'] = 'Communication or signaling'
                
                else:
                    interpretation['type'] = 'Animal Vocalization'
                    interpretation['meaning'] = 'Unknown purpose'
                
                # Calculate confidence in interpretation
                # This is simplified - real systems would use ML models
                # Higher confidence for more distinctive patterns
                if has_rapid_modulation and has_pitch_trends:
                    interp_confidence = 75
                elif has_rapid_modulation or has_pitch_trends:
                    interp_confidence = 60
                else:
                    interp_confidence = 40
                
                # Display interpretation
                interp_df = pd.DataFrame({
                    'Feature': ['Vocalization Type', 'Interpreted Meaning', 'Confidence'],
                    'Value': [interpretation.get('type', 'Unknown'), 
                             interpretation.get('meaning', 'Unknown'),
                             f"{interp_confidence}%"]
                })
                display(interp_df)
                
                # Show detected acoustic features
                features_df = pd.DataFrame({
                    'Feature': ['Mean Pitch', 'Pitch Range', 'Spectral Centroid', 'Rapid Modulation'],
                    'Value': [f"{np.mean(segment_pitches_values):.2f} Hz" if len(segment_pitches_values) > 0 else "Unknown",
                             f"{np.min(segment_pitches_values):.2f} - {np.max(segment_pitches_values):.2f} Hz" if len(segment_pitches_values) > 0 else "Unknown",
                             f"{np.mean(segment_spectral_centroid):.2f} Hz",
                             "Yes" if has_rapid_modulation else "No"]
                })
                display(features_df)
            
            else:
                print("Segment too short for analysis")
    
    else:
        printmd("No distinct communication events detected. Try adjusting sensitivity or analyzing a different audio sample.")
    
    return identified_species

# Form for URL input
from ipywidgets import widgets
from IPython.display import display

printmd("## URL Input Form")
printmd("Enter a direct URL to an audio file or use one of the sample URLs above.")

url_input = widgets.Text(
    value='',
    placeholder='Enter direct URL to audio file (.mp3, .wav)',
    description='Audio URL:',
    disabled=False,
    style={'description_width': 'initial'},
    layout={'width': '80%'}
)

species_dropdown = widgets.Dropdown(
    options=[('Auto-detect', None), ('Wolf', 'wolf'), ('Dolphin', 'dolphin'), 
             ('Chimpanzee', 'chimpanzee'), ('Bird', 'bird')],
    value=None,
    description='Species hint:',
    disabled=False,
    style={'description_width': 'initial'}
)

analyze_button = widgets.Button(
    description='Analyze Audio',
    disabled=False,
    button_style='primary',
    tooltip='Click to analyze the audio from URL',
    icon='check'
)

output = widgets.Output()

def on_analyze_button_clicked(b):
    with output:
        output.clear_output()
        if url_input.value:
            analyze_audio(url=url_input.value, species_hint=species_dropdown.value)
        else:
            print("Please enter a URL")

analyze_button.on_click(on_analyze_button_clicked)

display(url_input)
display(species_dropdown)
display(analyze_button)
display(output)

# Upload form for local files
printmd("## Or Upload an Audio File")

upload = widgets.FileUpload(
    accept='.mp3,.wav',
    multiple=False,
    description='Upload Audio:',
    style={'description_width': 'initial'},
    layout={'width': '80%'}
)

species_dropdown2 = widgets.Dropdown(
    options=[('Auto-detect', None), ('Wolf', 'wolf'), ('Dolphin', 'dolphin'), 
             ('Chimpanzee', 'chimpanzee'), ('Bird', 'bird')],
    value=None,
    description='Species hint:',
    disabled=False,
    style={'description_width': 'initial'}
)

analyze_upload_button = widgets.Button(
    description='Analyze Uploaded File',
    disabled=False,
    button_style='primary',
    tooltip='Click to analyze the uploaded audio file',
    icon='check'
)

upload_output = widgets.Output()

def on_analyze_upload_clicked(b):
    with upload_output:
        upload_output.clear_output()
        if upload.value:
            # Get the uploaded file
            file_list = list(upload.value.keys())
            if len(file_list) > 0:
                file_name = file_list[0]
                content = upload.value[file_name]['content']
                
                # Save to a temporary file
                with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file_name)[1]) as tmp:
                    tmp.write(content)
                    tmp_path = tmp.name
                
                # Analyze the file
                analyze_audio(file_path=tmp_path, species_hint=species_dropdown2.value)
                
                # Clean up
                os.unlink(tmp_path)
        else:
            print("Please upload an audio file")

analyze_upload_button.on_click(on_analyze_upload_clicked)

display(upload)
display(species_dropdown2)
display(analyze_upload_button)
display(upload_output)

# Display a note about the limitations
printmd("""
## Important Notes

1. **Audio Quality**: The analysis works best with clear recordings with minimal background noise.
2. **Species Identification**: This is a simplified approach to species identification and vocalization analysis. Real-world systems would use more sophisticated ML models trained on large datasets.
3. **Interpretation Limitations**: The "meaning" interpretations are based on general knowledge about animal communication and should be considered approximate.

## How This Works

This notebook performs the following analysis:
- Extracts the audio waveform and creates visualizations
- Generates spectrograms to visualize frequency components
- Identifies potential species based on acoustic features
- Detects individual communication events
- Analyzes each event for specific vocalization types
- Provides interpretations based on patterns in the audio

For a more accurate analysis, real bioacoustic research would:
- Use labeled datasets of known vocalizations
- Apply machine learning classification models
- Consider contextual information from observations
- Compare results with ethological research
""")

# Example usage code
printmd("""
## Example: Running the analyzer on a sample

```python
# You can use either approach:

# 1. From a URL
analyze_audio(url="https://cdn.freesound.org/previews/257/257855_4843504-lq.mp3", species_hint="dolphin")

# 2. From an uploaded file
# (Use the upload widget above)
```
""")
