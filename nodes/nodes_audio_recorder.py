import torch
import torchaudio
import sounddevice as sd
import numpy as np
import os
import time
import folder_paths

class AudioRecorder:
    @classmethod
    def INPUT_TYPES(cls):
        devices = sd.query_devices()
        input_devices = [d['name'] for d in devices if d['max_input_channels'] > 0]
        return {
            "required": {
                "device": (input_devices, {"default": input_devices[0] if input_devices else "Default"}),
                "duration": ("INT", {
                    "default": 10,
                    "min": 1,
                    "max": 120,
                    "step": 1,
                    "display": "slider"
                }),
                "sample_rate": ("INT", {
                    "default": 44100,
                    "min": 8000,
                    "max": 96000,
                    "step": 100,
                    "display": "number"
                }),
                "trigger": ("INT", {"default": 0})  # Move back to required but keep it hidden in JS
            }
        }

    def __init__(self):
        self.reset_state()

    def reset_state(self):
        """Clean up all old recordings"""
        # Clean up temp directory from our recordings
        if hasattr(self, 'temp_dir'):
            for filename in os.listdir(self.temp_dir):
                if filename.startswith("recorded_audio_") and filename.endswith(".wav"):
                    try:
                        os.remove(os.path.join(self.temp_dir, filename))
                        print(f"Cleaned up old recording: {filename}")
                    except Exception as e:
                        print(f"Error removing file {filename}: {e}")

        self.audio_data = []
        self.temp_dir = folder_paths.get_temp_directory()
        self.recorded_file = None
        self.last_trigger = -1

    RETURN_TYPES = ("AUDIO",)
    FUNCTION = "record"
    CATEGORY = "audio"

    def record(self, device, duration, sample_rate, trigger):
        try:
            if trigger != self.last_trigger:
                self.last_trigger = trigger
                print(f"\nStarting new recording...")
                print(f"Settings: device={device}, duration={duration}s, rate={sample_rate}")
                
                try:
                    print("Opening audio stream...")
                    with sd.InputStream(
                        device=None,
                        channels=1,
                        samplerate=sample_rate,
                    ) as stream:
                        print("Recording in progress...")
                        frames = int(duration * sample_rate)
                        audio_data = stream.read(frames)[0]
                        self.audio_data = audio_data
                    
                    print("Processing recording...")
                    audio_tensor = torch.from_numpy(self.audio_data).float().t()
                    temp_file = os.path.join(self.temp_dir, f"recorded_audio_{int(time.time())}.wav")
                    torchaudio.save(temp_file, audio_tensor, sample_rate)
                    self.recorded_file = temp_file
                    print(f"Recording saved to: {temp_file}")
                    
                    waveform, sr = torchaudio.load(self.recorded_file)
                    return ({"waveform": waveform.unsqueeze(0), "sample_rate": sr},)
                    
                except Exception as e:
                    print(f"Error recording audio: {e}")
                    import traceback
                    traceback.print_exc()
            
            elif self.recorded_file and os.path.exists(self.recorded_file):
                waveform, sr = torchaudio.load(self.recorded_file)
                return ({"waveform": waveform.unsqueeze(0), "sample_rate": sr},)
            
        except Exception as e:
            print(f"Error in record method: {e}")
            import traceback
            traceback.print_exc()
        
        return ({"waveform": torch.zeros(1, 1, 1), "sample_rate": sample_rate},)

NODE_CLASS_MAPPINGS = {
    "AudioRecorder": AudioRecorder
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AudioRecorder": "🎤 Audio Recorder"
}