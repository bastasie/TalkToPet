

## Installation

This tool is designed to run in Google Colab, which provides all necessary dependencies pre-installed.

1. Open Google Colab (https://colab.research.google.com/)
2. Create a new notebook
3. Copy the entire code into a cell
4. Run the cell

If you wish to run locally, install the required dependencies:

```bash
pip install numpy pandas matplotlib librosa scipy requests ipython ipywidgets
```

## Usage

### In Google Colab

1. Run the cell containing the code
2. Choose one of the following methods to analyze audio:

   **Method 1: Analyze from URL**
   - Enter a direct URL to an MP3 or WAV file in the URL input field
   - Optionally select a species hint from the dropdown
   - Click "Analyze Audio"

   **Method 2: Analyze uploaded file**
   - Click the "Upload Audio" button
   - Select an MP3 or WAV file from your computer
   - Optionally select a species hint from the dropdown
   - Click "Analyze Uploaded File"

### Programmatic Usage

```python
# Analyze audio from URL
analyze_audio(url="https://example.com/animal_sound.mp3", species_hint="wolf")

# Analyze audio from file
analyze_audio(file_path="path/to/local/file.wav")
```

## Sample Audio URLs

The tool provides several example audio links for testing:

- Wolf Howl: https://freesound.org/data/previews/398/398713_5121236-lq.mp3
- Dolphin: https://cdn.freesound.org/previews/257/257855_4843504-lq.mp3
- Chimp Calls: https://cdn.freesound.org/previews/338/338428_5865517-lq.mp3
- Bird Songs: https://cdn.freesound.org/previews/561/561320_2160092-lq.mp3

## Analysis Components

The analyzer performs several types of analysis:

1. **Basic Audio Information**: Duration, sample rate, playback capabilities
2. **Waveform Analysis**: Temporal visualization of amplitude variations
3. **Spectrogram Analysis**: Time-frequency representation of audio
4. **Mel Spectrogram**: Perceptually-weighted frequency representation
5. **Pitch Tracking**: Fundamental frequency estimation
6. **Spectral Feature Extraction**: Centroid, bandwidth, zero-crossing rate
7. **Species Identification**: Classification based on acoustic properties
8. **Event Detection**: Isolation of distinct communication segments
9. **Event Analysis**: Detailed examination of each detected vocalization

## Example Output

For each analysis, the tool provides:

- **Visualizations**: Waveforms, spectrograms, and event detection highlights
- **Species Information**: Identified species with confidence score
- **Communication Events**: Timestamped list of detected vocalizations
- **Event Analysis**: For each detected event:
  - Audio playback of the isolated event
  - Spectrogram visualization
  - Classification of vocalization type
  - Interpretation of potential meaning
  - Confidence assessment
  - Acoustic feature measurements

## Limitations

- **Simplified Approach**: This tool uses basic rule-based methods rather than sophisticated machine learning models used in professional research
- **Interpretation Confidence**: Meaning interpretations are approximations based on general knowledge about animal communication
- **Species Coverage**: Currently supports limited species profiles (wolf, dolphin, chimpanzee, bird)
- **Audio Quality**: Performance is highly dependent on recording quality and noise levels

## Technical Details

The analyzer employs several signal processing techniques:

- **Fourier Transform**: Converts time-domain signals to frequency domain
- **Short-Time Fourier Transform (STFT)**: Generates spectrograms
- **Mel Filtering**: Applies perceptual weighting to frequency analysis
- **Energy-Based Segmentation**: Detects communication events using amplitude thresholds
- **Feature Extraction**: Calculates acoustic parameters like spectral centroid and bandwidth
- **Pitch Tracking**: Estimates fundamental frequency using harmonic detection

## Future Development

Potential enhancements for future versions include:

- Implementing deep learning models for improved classification and interpretation
- Expanding the species database to cover more animal groups
- Adding support for underwater recordings and non-vocal communication
- Developing a standalone application that doesn't require Google Colab

## Contributing

Contributions to improve the analyzer are welcome. Please feel free to submit pull requests or open issues to discuss potential enhancements.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use this tool in your research, please cite it as:

```
Animal Vocalization Analyzer. (2025). GitHub Repository. 
https://github.com/Bastasie/TalkToPet 
```

## Acknowledgments

- The development of this tool was inspired by recent advances in bioacoustics research
- Sample audio files provided by Freesound.org contributors
- Built using the librosa audio processing library
