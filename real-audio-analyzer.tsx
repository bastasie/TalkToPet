import React, { useState, useEffect, useRef } from 'react';

const RealTimeAudioAnalyzer = () => {
  const [isPlaying, setIsPlaying] = useState(false);
  const [selectedDemo, setSelectedDemo] = useState(null);
  const [analysisResults, setAnalysisResults] = useState(null);
  const [error, setError] = useState('');
  const [loadingAudio, setLoadingAudio] = useState(false);
  const [frequencyData, setFrequencyData] = useState(null);
  
  const audioRef = useRef(null);
  const audioContextRef = useRef(null);
  const analyzerRef = useRef(null);
  const sourceRef = useRef(null);
  const animationRef = useRef(null);
  const canvasRef = useRef(null);
  
  // Demo audio files with embedded animal sounds
  const demoFiles = {
    'Chimpanzee': '/www.storyblocks.com/audio/stock/chimps-rekxust2uwsk0wxxpx5.html',
    'Dolphin': '/www.storyblocks.com/audio/stock/dolphin-chirping-hlmqcbp3uvhk0wxxrfb.html',
    'Wolf': '/www.storyblocks.com/audio/stock/bird-ambience-bgqxuqo2idhk0wxogg2.html',
    'Bird': '/www.storyblocks.com/audio/stock/wolf-sgnmdlan8dsk0wxy1hb.html',
    'Elephant': '/www.storyblocks.com/audio/stock/elephant-2-re0bjbp38psk0wxxs1w.html'
    


https://www.storyblocks.com/audio/stock/bird-ambience-bgqxuqo2idhk0wxogg2.html
use these audio samples they don't need api
  };
  
  // Simulated species identification based on spectral content
  const identifySpecies = (frequencyData) => {
    // In a real implementation, this would use machine learning
    // to identify the species based on acoustic features
    
    if (!frequencyData) return 'Unknown';
    
    // For demo purposes, we'll return the selected demo species
    return selectedDemo || 'Unknown';
  };

  // Process audio and extract features for analysis
  const processAudio = () => {
    if (!analyzerRef.current || !frequencyData) return null;
    
    // Calculate spectral properties
    const sampleRate = audioContextRef.current?.sampleRate || 44100;
    const bufferLength = analyzerRef.current.frequencyBinCount;
    const binSize = sampleRate / (bufferLength * 2);
    
    // Find dominant frequencies
    let maxIndex = 0;
    let maxValue = 0;
    let totalEnergy = 0;
    
    for (let i = 0; i < frequencyData.length; i++) {
      const value = frequencyData[i];
      totalEnergy += value;
      
      if (value > maxValue) {
        maxValue = value;
        maxIndex = i;
      }
    }
    
    const dominantFrequency = maxIndex * binSize;
    const average = totalEnergy / frequencyData.length;
    
    // Find frequency bands energy distribution
    const bands = [
      { name: 'Low', min: 0, max: 500, energy: 0 },
      { name: 'Mid-low', min: 500, max: 1000, energy: 0 },
      { name: 'Mid', min: 1000, max: 2000, energy: 0 },
      { name: 'Mid-high', min: 2000, max: 5000, energy: 0 },
      { name: 'High', min: 5000, max: 20000, energy: 0 }
    ];
    
    for (let i = 0; i < frequencyData.length; i++) {
      const freq = i * binSize;
      const energy = frequencyData[i];
      
      for (const band of bands) {
        if (freq >= band.min && freq <= band.max) {
          band.energy += energy;
        }
      }
    }
    
    // Normalize band energies
    const maxEnergy = Math.max(...bands.map(band => band.energy));
    bands.forEach(band => {
      band.relativeEnergy = band.energy / maxEnergy;
    });
    
    // Generate analysis based on spectral features
    const species = identifySpecies(frequencyData);
    const context = determineContext(bands, dominantFrequency);
    
    return {
      species,
      context,
      spectralFeatures: {
        dominantFrequency: Math.round(dominantFrequency),
        spectralCentroid: Math.round(calculateSpectralCentroid(frequencyData, binSize)),
        spectralFlatness: calculateSpectralFlatness(frequencyData),
        spectralSlope: calculateSpectralSlope(frequencyData)
      },
      frequencyBands: bands,
      signals: interpretSignals(species, context, bands, dominantFrequency)
    };
  };
  
  // Calculate spectral centroid (brightness)
  const calculateSpectralCentroid = (frequencyData, binSize) => {
    let numerator = 0;
    let denominator = 0;
    
    for (let i = 0; i < frequencyData.length; i++) {
      const frequency = i * binSize;
      const amplitude = frequencyData[i];
      
      numerator += frequency * amplitude;
      denominator += amplitude;
    }
    
    return denominator === 0 ? 0 : numerator / denominator;
  };
  
  // Calculate spectral flatness (tonal vs. noisy)
  const calculateSpectralFlatness = (frequencyData) => {
    let geometricMean = 0;
    let arithmeticMean = 0;
    
    // Add small value to avoid log(0)
    const data = frequencyData.map(v => v + 0.00001);
    
    // Calculate geometric mean
    let sumOfLogs = 0;
    for (let i = 0; i < data.length; i++) {
      sumOfLogs += Math.log(data[i]);
      arithmeticMean += data[i];
    }
    
    geometricMean = Math.exp(sumOfLogs / data.length);
    arithmeticMean = arithmeticMean / data.length;
    
    return geometricMean / arithmeticMean;
  };
  
  // Calculate spectral slope (distribution of energy)
  const calculateSpectralSlope = (frequencyData) => {
    const n = frequencyData.length;
    let sumX = 0;
    let sumY = 0;
    let sumXY = 0;
    let sumXX = 0;
    
    for (let i = 0; i < n; i++) {
      const x = i;
      const y = frequencyData[i];
      
      sumX += x;
      sumY += y;
      sumXY += x * y;
      sumXX += x * x;
    }
    
    return (n * sumXY - sumX * sumY) / (n * sumXX - sumX * sumX);
  };
  
  // Determine the communication context
  const determineContext = (bands, dominantFrequency) => {
    // This would be a much more sophisticated analysis in reality
    if (dominantFrequency < 500) return 'social long-distance';
    if (dominantFrequency > 5000) return 'alert';
    
    // Check energy distribution
    const lowEnergy = bands[0].relativeEnergy + bands[1].relativeEnergy;
    const highEnergy = bands[3].relativeEnergy + bands[4].relativeEnergy;
    
    if (lowEnergy > 0.7) return 'boredom';
    if (highEnergy > 0.7) return 'excitement/novelty';
    
    return 'social communication';
  };
  
  // Interpret the signals based on spectral analysis
  const interpretSignals = (species, context, bands, dominantFrequency) => {
    // This would use a database of known vocalizations in reality
    const signals = [];
    
    // Add generic interpretation based on spectral features
    if (context === 'boredom') {
      signals.push({
        type: dominantFrequency < 1000 ? 'Low frequency vocalization' : 'Mid-range call',
        intensity: bands[2].relativeEnergy > 0.7 ? 'high' : 'medium',
        decodedMeaning: 'Expression of dissatisfaction / seeking stimulation',
        confidence: 70
      });
      
      // Add species-specific interpretation
      if (species === 'Chimpanzee') {
        signals.push({
          type: 'Soft food bark',
          intensity: 'medium',
          decodedMeaning: 'I want something to do / Pay attention to me',
          confidence: 85
        });
      } else if (species === 'Dolphin') {
        signals.push({
          type: 'Repetitive whistle',
          intensity: 'medium',
          decodedMeaning: 'Social contact seeking / Where is everyone?',
          confidence: 82
        });
      }
    } else if (context === 'alert') {
      signals.push({
        type: 'High frequency alarm call',
        intensity: 'high',
        decodedMeaning: 'Alert! Something unusual detected',
        confidence: 88
      });
    } else {
      signals.push({
        type: dominantFrequency < 2000 ? 'Social contact call' : 'Information sharing vocalization',
        intensity: 'medium',
        decodedMeaning: 'General social communication / group coordination',
        confidence: 75
      });
    }
    
    return signals;
  };

  // Setup audio context and analyzer
  const setupAudioAnalysis = async () => {
    // Reset any previous audio context
    if (audioContextRef.current) {
      audioContextRef.current.close();
    }
    
    // Create new audio context
    const audioContext = new (window.AudioContext || window.webkitAudioContext)();
    audioContextRef.current = audioContext;
    
    // Create analyzer node
    const analyzer = audioContext.createAnalyser();
    analyzer.fftSize = 2048;
    analyzer.smoothingTimeConstant = 0.85;
    analyzerRef.current = analyzer;
    
    // Connect audio element to analyzer
    if (audioRef.current) {
      const source = audioContext.createMediaElementSource(audioRef.current);
      source.connect(analyzer);
      analyzer.connect(audioContext.destination);
      sourceRef.current = source;
    }
  };
  
  // Handle demo selection
  const handleSelectDemo = async (demo) => {
    setSelectedDemo(demo);
    setIsPlaying(false);
    setLoadingAudio(true);
    setError('');
    
    try {
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current);
      }
      
      if (audioRef.current) {
        audioRef.current.pause();
        audioRef.current.src = demoFiles[demo];
        
        // Wait for audio to be ready
        await new Promise((resolve, reject) => {
          audioRef.current.oncanplaythrough = resolve;
          audioRef.current.onerror = reject;
        });
        
        await setupAudioAnalysis();
        setLoadingAudio(false);
      }
    } catch (err) {
      setError(`Error loading audio: ${err.message}`);
      setLoadingAudio(false);
    }
  };
  
  // Toggle play/pause
  const togglePlay = () => {
    if (!audioRef.current) return;
    
    if (isPlaying) {
      audioRef.current.pause();
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current);
      }
    } else {
      audioRef.current.play();
      audioContextRef.current.resume();
      updateAnalyzer();
    }
    
    setIsPlaying(!isPlaying);
  };
  
  // Update analyzer and draw spectrogram
  const updateAnalyzer = () => {
    if (!analyzerRef.current || !canvasRef.current) {
      animationRef.current = requestAnimationFrame(updateAnalyzer);
      return;
    }
    
    const bufferLength = analyzerRef.current.frequencyBinCount;
    const dataArray = new Uint8Array(bufferLength);
    analyzerRef.current.getByteFrequencyData(dataArray);
    
    // Store frequency data for analysis
    setFrequencyData(Array.from(dataArray));
    
    // Draw spectrogram
    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    const width = canvas.width;
    const height = canvas.height;
    
    // Shift existing content left
    const imageData = ctx.getImageData(10, 0, width - 10, height);
    ctx.putImageData(imageData, 0, 0);
    
    // Draw new column
    for (let i = 0; i < bufferLength; i++) {
      const v = dataArray[i];
      const y = height - (i / bufferLength) * height;
      
      // Create gradient color based on intensity
      const intensity = v / 255;
      const r = Math.floor(intensity * 255);
      const g = Math.floor(50 * intensity);
      const b = Math.floor(255 * (1 - intensity));
      
      ctx.fillStyle = `rgb(${r}, ${g}, ${b})`;
      ctx.fillRect(width - 2, y, 2, 2);
    }
    
    // Process audio and update results
    const results = processAudio();
    if (results) {
      setAnalysisResults(results);
    }
    
    animationRef.current = requestAnimationFrame(updateAnalyzer);
  };
  
  // Clear canvas when component unmounts
  useEffect(() => {
    return () => {
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current);
      }
      
      if (audioContextRef.current) {
        audioContextRef.current.close();
      }
    };
  }, []);

  // Reset and initialize canvas when it changes
  useEffect(() => {
    if (canvasRef.current) {
      const canvas = canvasRef.current;
      const ctx = canvas.getContext('2d');
      ctx.fillStyle = '#000000';
      ctx.fillRect(0, 0, canvas.width, canvas.height);
    }
  }, [canvasRef.current]);
  
  // Format confidence as colored badge
  const getConfidenceBadge = (confidence) => {
    let color = 'bg-red-500';
    if (confidence >= 80) color = 'bg-green-500';
    else if (confidence >= 60) color = 'bg-yellow-500';
    
    return (
      <span className={`${color} text-white text-xs px-2 py-1 rounded-full`}>
        {confidence}%
      </span>
    );
  };

  return (
    <div className="max-w-4xl mx-auto p-4">
      <h1 className="text-2xl font-bold mb-4">Real-Time Animal Audio Analyzer</h1>
      
      <div className="bg-blue-50 p-4 rounded-lg mb-6">
        <p className="text-sm">
          This tool performs real-time spectral analysis on animal vocalization audio.
          Select a demo file to analyze its frequency content and decode potential meanings.
        </p>
      </div>
      
      <div className="mb-6">
        <div className="flex flex-wrap gap-2 mb-4">
          <span className="font-medium mt-1">Select demo:</span>
          {Object.keys(demoFiles).map((demo) => (
            <button
              key={demo}
              onClick={() => handleSelectDemo(demo)}
              className={`px-3 py-1 rounded ${selectedDemo === demo ? 'bg-blue-600 text-white' : 'bg-gray-200 hover:bg-gray-300'}`}
            >
              {demo}
            </button>
          ))}
        </div>
        
        {loadingAudio ? (
          <div className="flex items-center justify-center p-8">
            <svg className="animate-spin h-8 w-8 text-blue-500" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
              <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
              <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
            </svg>
            <span className="ml-3">Loading audio...</span>
          </div>
        ) : selectedDemo ? (
          <div className="border border-gray-300 rounded-lg p-4">
            <div className="flex items-center mb-4">
              <button
                onClick={togglePlay}
                className={`${isPlaying ? 'bg-red-600' : 'bg-green-600'} text-white font-medium py-2 px-4 rounded-lg mr-4`}
              >
                {isPlaying ? 'Pause Analysis' : 'Start Analysis'}
              </button>
              <div className="text-gray-700">
                Selected: <span className="font-medium">{selectedDemo}</span>
              </div>
            </div>
            
            <div className="mb-4 bg-black rounded-lg overflow-hidden">
              <canvas ref={canvasRef} width="600" height="200" className="w-full"></canvas>
              <div className="flex justify-between text-xs text-gray-400 px-2">
                <span>Time →</span>
                <span>Real-time spectrogram (frequency vs. time)</span>
              </div>
            </div>
            
            <audio ref={audioRef} className="hidden" loop />
            
            {error && <div className="text-red-600 mt-2">{error}</div>}
          </div>
        ) : (
          <div className="text-center p-8 bg-gray-100 rounded-lg">
            <p className="text-gray-500">Please select a demo audio file to begin analysis</p>
          </div>
        )}
      </div>
      
      {analysisResults && (
        <div className="border border-gray-300 rounded-lg overflow-hidden">
          <div className="bg-gray-100 p-4 border-b">
            <h2 className="text-xl font-bold">Analysis Results</h2>
          </div>
          
          <div className="p-4">
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <div>
                <div className="mb-4">
                  <h3 className="font-bold text-lg mb-2">Spectral Analysis</h3>
                  <div className="bg-white p-3 rounded border border-gray-200">
                    <div className="grid grid-cols-2 gap-y-2">
                      <div className="text-gray-600">Species:</div>
                      <div className="font-medium">{analysisResults.species}</div>
                      
                      <div className="text-gray-600">Context:</div>
                      <div className="font-medium">{analysisResults.context}</div>
                      
                      <div className="text-gray-600">Dominant Frequency:</div>
                      <div className="font-medium">{analysisResults.spectralFeatures.dominantFrequency} Hz</div>
                      
                      <div className="text-gray-600">Spectral Centroid:</div>
                      <div className="font-medium">{analysisResults.spectralFeatures.spectralCentroid} Hz</div>
                    </div>
                  </div>
                </div>
                
                <div>
                  <h3 className="font-bold text-lg mb-2">Frequency Band Distribution</h3>
                  <div className="space-y-2">
                    {analysisResults.frequencyBands.map((band, index) => (
                      <div key={index} className="flex items-center">
                        <div className="w-24 text-sm">{band.name} ({band.min}-{band.max} Hz):</div>
                        <div className="flex-grow bg-gray-200 rounded-full h-4 overflow-hidden">
                          <div
                            className="bg-blue-600 h-4"
                            style={{ width: `${band.relativeEnergy * 100}%` }}
                          ></div>
                        </div>
                        <div className="w-12 text-right text-sm ml-2">
                          {Math.round(band.relativeEnergy * 100)}%
                        </div>
                      </div>
                    ))}
                  </div>
                  <div className="text-xs text-gray-500 mt-1">
                    Energy distribution across frequency bands
                  </div>
                </div>
              </div>
              
              <div>
                <h3 className="font-bold text-lg mb-2">Decoded Communications</h3>
                <div className="space-y-3">
                  {analysisResults.signals.map((signal, index) => (
                    <div key={index} className="border border-gray-200 rounded bg-white p-3">
                      <div className="flex justify-between items-start">
                        <span className="text-xs bg-blue-100 rounded px-2 py-1">{signal.type}</span>
                        <span className="text-xs bg-gray-200 rounded px-2 py-1">Intensity: {signal.intensity}</span>
                      </div>
                      
                      <div className="mt-2 bg-blue-50 p-3 rounded">
                        <div className="flex justify-between">
                          <span className="font-medium">Decoded meaning:</span>
                          {getConfidenceBadge(signal.confidence)}
                        </div>
                        <div className="mt-1 text-blue-800">{signal.decodedMeaning}</div>
                      </div>
                    </div>
                  ))}
                </div>
                
                <div className="mt-4 bg-yellow-50 p-3 rounded text-sm">
                  <p className="font-medium mb-1">Interpretation:</p>
                  <p>
                    {analysisResults.context === 'boredom' 
                      ? `The spectral analysis indicates vocalization patterns consistent with boredom or understimulation in ${analysisResults.species}. The relatively low frequency content and repetitive patterns suggest the animal is attempting to solicit attention or activity.`
                      : analysisResults.context === 'alert'
                      ? `The high-frequency content and energy distribution pattern is consistent with alert or alarm calls in ${analysisResults.species}. This suggests the animal is communicating about something novel or potentially concerning in the environment.`
                      : `The spectral features suggest this is a ${analysisResults.context} vocalization from ${analysisResults.species}. The frequency distribution and modulation patterns align with known communication patterns for this context.`}
                  </p>
                </div>
              </div>
            </div>
          </div>
        </div>
      )}
      
      <div className="mt-6 text-xs text-gray-500 border-t pt-4">
        <p>
          This analyzer performs real spectral analysis on audio using the Web Audio API. In a complete system, 
          this would be combined with machine learning models trained on thousands of labeled animal vocalizations
          to identify species, contexts, and specific communication types. The frequency analysis shown here
          is real, but the interpretations are simulated as they would require extensive training data.
        </p>
      </div>
    </div>
  );
};

export default RealTimeAudioAnalyzer;
