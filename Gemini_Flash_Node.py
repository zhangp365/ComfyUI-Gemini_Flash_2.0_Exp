# Gemini_Flash_Node.py
import os
import json
import google.generativeai as genai
from io import BytesIO
from PIL import Image
import torch
import torchaudio
import numpy as np
from contextlib import contextmanager

p = os.path.dirname(os.path.realpath(__file__))

def get_config():
    try:
        config_path = os.path.join(p, 'config.json')
        with open(config_path, 'r') as f:  
            config = json.load(f)
        return config
    except:
        return {}

def save_config(config):
    config_path = os.path.join(p, 'config.json')
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=4)

@contextmanager
def temporary_env_var(key: str, new_value):
    old_value = os.environ.get(key)
    if new_value is not None:
        os.environ[key] = new_value
    elif key in os.environ:
        del os.environ[key]
    try:
        yield
    finally:
        if old_value is not None:
            os.environ[key] = old_value
        elif key in os.environ:
            del os.environ[key]

class ChatHistory:
    def __init__(self):
        self.messages = []
    
    def add_message(self, role, content):
        if isinstance(content, list):
            content = " ".join(str(item) for item in content if isinstance(item, str))
        self.messages.append({"role": role, "content": content})
    
    def get_formatted_history(self):
        formatted = "\n=== Chat History ===\n"
        for msg in self.messages:
            formatted += f"{msg['role'].upper()}: {msg['content']}\n"
        formatted += "=== End History ===\n"
        return formatted
    
    def get_messages_for_api(self):
        api_messages = []
        for msg in self.messages:
            if isinstance(msg["content"], str):
                api_messages.append({
                    "role": msg["role"],
                    "parts": [{"text": msg["content"]}]
                })
        return api_messages
    
    def clear(self):
        self.messages = []

class Gemini_Flash_200_Exp:
    def __init__(self, api_key=None, proxy=None):
        config = get_config()
        self.api_key = api_key or config.get("GEMINI_API_KEY")
        self.proxy = proxy or config.get("PROXY")
        self.chat_history = ChatHistory()
        if self.api_key is not None:
            self.configure_genai()

    def configure_genai(self):
        genai.configure(api_key=self.api_key, transport='rest')

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"default": "Analyze the situation in details.", "multiline": True}),
                "input_type": (["text", "image", "video", "audio"], {"default": "text"}),
                "chat_mode": ("BOOLEAN", {"default": False}),
                "clear_history": ("BOOLEAN", {"default": False})
            },
            "optional": {
                "text_input": ("STRING", {"default": "", "multiline": True}),
                "image": ("IMAGE",),
                "video": ("IMAGE",),
                "audio": ("AUDIO",),
                "api_key": ("STRING", {"default": ""}),
                "proxy": ("STRING", {"default": ""}),
                "max_output_tokens": ("INT", {"default": 8192, "min": 1, "max": 8192}),
                "temperature": ("FLOAT", {"default": 0.4, "min": 0.0, "max": 1.0, "step": 0.1}),
                "structured_output": ("BOOLEAN", {"default": False})
                
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("generated_content",)
    FUNCTION = "generate_content"
    CATEGORY = "Gemini Flash 2.0 Experimental"

    def tensor_to_image(self, tensor):
        tensor = tensor.cpu()
        image_np = tensor.squeeze().mul(255).clamp(0, 255).byte().numpy()
        image = Image.fromarray(image_np, mode='RGB')
        return image

    def resize_image(self, image, max_size):
        width, height = image.size
        if width > height:
            if width > max_size:
                height = int(max_size * height / width)
                width = max_size
        else:
            if height > max_size:
                width = int(max_size * width / height)
                height = max_size
        return image.resize((width, height), Image.LANCZOS)

    def sample_video_frames(self, video_tensor, num_samples=6):
        """Sample frames evenly from video tensor"""
        if len(video_tensor.shape) != 4:
            return None

        total_frames = video_tensor.shape[0]
        if total_frames <= num_samples:
            indices = range(total_frames)
        else:
            indices = np.linspace(0, total_frames - 1, num_samples, dtype=int)

        frames = []
        for idx in indices:
            frame = self.tensor_to_image(video_tensor[idx])
            frame = self.resize_image(frame, 512)
            frames.append(frame)
        return frames

    def prepare_content(self, prompt, input_type, text_input, image, video, audio):
        if input_type == "text":
            text_content = prompt if not text_input else f"{prompt}\n{text_input}"
            return [{"text": text_content}]
                
        elif input_type == "image" and image is not None:
            pil_image = self.tensor_to_image(image)
            pil_image = self.resize_image(pil_image, 1024)
            # Convert image to bytes
            img_byte_arr = BytesIO()
            pil_image.save(img_byte_arr, format='PNG')
            img_byte_arr = img_byte_arr.getvalue()
            
            return [{
                "parts": [
                    {"text": prompt},
                    {
                        "inline_data": {
                            "mime_type": "image/png",
                            "data": img_byte_arr
                        }
                    }
                ]
            }]
                
        elif input_type == "video" and video is not None:
            # Handle video input (sequence of frames)
            frames = self.sample_video_frames(video)
            if frames:
                # Convert frames to proper format
                parts = [{"text": f"Analyzing video frames. {prompt}"}]
                for frame in frames:
                    # Convert each frame to bytes
                    img_byte_arr = BytesIO()
                    frame.save(img_byte_arr, format='PNG')
                    img_byte_arr = img_byte_arr.getvalue()
                    
                    parts.append({
                        "inline_data": {
                            "mime_type": "image/png",
                            "data": img_byte_arr
                        }
                    })
                return [{"parts": parts}]
            else:
                raise ValueError("Invalid video format")
                    
        elif input_type == "audio" and audio is not None:
            waveform = audio["waveform"]
            sample_rate = audio["sample_rate"]
            
            if waveform.dim() == 3:
                waveform = waveform.squeeze(0)
            elif waveform.dim() == 1:
                waveform = waveform.unsqueeze(0)
            
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)
            
            if sample_rate != 16000:
                waveform = torchaudio.functional.resample(waveform, sample_rate, 16000)
            
            buffer = BytesIO()
            torchaudio.save(buffer, waveform, 16000, format="WAV")
            audio_bytes = buffer.getvalue()
            
            return [{
                "parts": [
                    {"text": prompt},
                    {
                        "inline_data": {
                            "mime_type": "audio/wav",
                            "data": audio_bytes
                        }
                    }
                ]
            }]
        else:
            raise ValueError(f"Invalid or missing input for {input_type}")

    def generate_content(self, prompt, input_type, chat_mode=False, clear_history=False,
                        text_input=None, image=None, video=None, audio=None, 
                        api_key="", proxy="",
                        max_output_tokens=8192, temperature=0.4, structured_output=False):
        """Generate content using Gemini model with various input types."""
        
        # Set all safety settings to block_none by default
        safety_settings = [
            {"category": "harassment", "threshold": "NONE"},
            {"category": "hate_speech", "threshold": "NONE"},
            {"category": "sexually_explicit", "threshold": "NONE"},
            {"category": "dangerous_content", "threshold": "NONE"},
            {"category": "civic", "threshold": "NONE"}
        ]

        # Only update API key if explicitly provided in the node
        if api_key.strip():
            self.api_key = api_key
            save_config({"GEMINI_API_KEY": self.api_key, "PROXY": self.proxy})
            self.configure_genai()
        
        # Only update proxy if explicitly provided in the node    
        if proxy.strip():
            self.proxy = proxy
            save_config({"GEMINI_API_KEY": self.api_key, "PROXY": self.proxy})

        if not self.api_key:
            raise ValueError("API key not found in config.json or node input")

        if clear_history:
            self.chat_history.clear()

        model_name = 'models/gemini-2.0-flash-exp'
        model = genai.GenerativeModel(model_name)

        # Apply fixed safety settings to the model
        model.safety_settings = safety_settings

        generation_config = genai.types.GenerationConfig(
            max_output_tokens=max_output_tokens,
            temperature=temperature
        )

        with temporary_env_var('HTTP_PROXY', self.proxy), temporary_env_var('HTTPS_PROXY', self.proxy):
            try:
                if chat_mode:
                    # Special handling for chat mode
                    if input_type == "text":
                        text_content = prompt if not text_input else f"{prompt}\n{text_input}"
                        content = text_content
                    elif input_type == "image" and image is not None:
                        pil_image = self.tensor_to_image(image)
                        pil_image = self.resize_image(pil_image, 1024)
                        content = [prompt, pil_image]
                    elif input_type == "video" and video is not None:
                        if len(video.shape) == 4 and video.shape[0] > 1:
                            frame_count = video.shape[0]
                            frames = self.sample_video_frames(video)
                            if frames:
                                content = [f"This is a video with {frame_count} frames. {prompt}"] + frames
                            else:
                                raise ValueError("Error processing video frames")
                        else:
                            pil_image = self.tensor_to_image(video.squeeze(0) if len(video.shape) == 4 else video)
                            pil_image = self.resize_image(pil_image, 1024)
                            content = [f"This is a single frame from a video. {prompt}", pil_image]
                    elif input_type == "audio" and audio is not None:
                        waveform = audio["waveform"]
                        sample_rate = audio["sample_rate"]
                        
                        if waveform.dim() == 3:
                            waveform = waveform.squeeze(0)
                        elif waveform.dim() == 1:
                            waveform = waveform.unsqueeze(0)
                        
                        if waveform.shape[0] > 1:
                            waveform = torch.mean(waveform, dim=0, keepdim=True)
                        
                        if sample_rate != 16000:
                            waveform = torchaudio.functional.resample(waveform, sample_rate, 16000)
                        
                        buffer = BytesIO()
                        torchaudio.save(buffer, waveform, 16000, format="WAV")
                        audio_bytes = buffer.getvalue()
                        
                        content = [prompt, {"mime_type": "audio/wav", "data": audio_bytes}]
                    else:
                        raise ValueError(f"Invalid or missing input for {input_type}")

                    # Initialize chat and send message
                    chat = model.start_chat(history=self.chat_history.get_messages_for_api())
                    response = chat.send_message(content, generation_config=generation_config)
                    
                    # Add to history and get formatted output
                    if isinstance(content, list):
                        history_content = prompt
                    else:
                        history_content = content
                        
                    self.chat_history.add_message("user", history_content)
                    self.chat_history.add_message("assistant", response.text)
                    
                    # Only show the chat history
                    generated_content = self.chat_history.get_formatted_history()
                else:
                    # Non-chat mode uses the prepare_content method
                    content_parts = self.prepare_content(prompt, input_type, text_input, image, video, audio)
                    if structured_output:
                        if isinstance(content_parts, list) and len(content_parts) > 0:
                            if "parts" in content_parts[0]:
                                for part in content_parts[0]["parts"]:
                                    if "text" in part:
                                        part["text"] = f"Please provide the response in a structured format. {part['text']}"
                    
                    response = model.generate_content(content_parts, generation_config=generation_config)
                    generated_content = response.text

            except Exception as e:
                generated_content = f"Error: {str(e)}"
        
        return (generated_content,)
        
NODE_CLASS_MAPPINGS = {
    "Gemini_Flash_200_Exp": Gemini_Flash_200_Exp,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Gemini_Flash_200_Exp": "Gemini Flash 2.0 Experimental",
}