# ComfyUI-Gemini_Flash_2.0_Exp

A ComfyUI custom node that integrates Google's Gemini Flash 2.0 Experimental model, enabling multimodal analysis of text, images, video frames, and audio directly within ComfyUI workflows.

## Features

- Multimodal input support:
  - Text analysis
  - Image analysis
  - Video frame analysis
  - Audio analysis
- Chat mode with conversation history
- Structured output option
- Temperature and token limit controls
- Proxy support
- Configurable API settings via config.json

## Installation

1. Clone this repository into your ComfyUI custom_nodes folder:
```bash
cd ComfyUI/custom_nodes
git clone https://github.com/yourusername/ComfyUI-Gemini_Flash_2.0_Exp
```

2. Install required dependencies:
```bash
pip install google.generativeai
pip install pillow
pip install torchaudio
```

3. Get your free API key from Google AI Studio:
   - Visit [Google AI Studio](https://aistudio.google.com/prompts/new_chat)
   - Log in with your Google account
   - Click on "Get API key" or go to settings
   - Create a new API key
   - Copy the API key for use in config.json

4. Set up your API key in the `config.json` file (will be created automatically on first run)

## Configuration

### API Key Setup

The `config.json` file in the node folder contains all configuration settings:
```json
{
    "GEMINI_API_KEY": "your_api_key_here",
    "PROXY": "",
    "MODEL_NAME": "models/gemini-2.0-flash-exp",
    "RPM_LIMIT": 10,
    "TPM_LIMIT": 4000000,
    "RPD_LIMIT": 1500,
    "DEFAULT_CHAT_MODE": false
}
```

### Proxy Setup (Optional)

If you need to use a proxy:
- Set it in config.json under the "PROXY" field
- Format: "http://your-proxy:port" or "socks5://your-proxy:port"

## Node Inputs

### Required Inputs:
- **prompt**: Main text prompt for analysis or generation
- **input_type**: Select from ["text", "image", "video", "audio"]
- **chat_mode**: Boolean to enable/disable chat functionality
- **clear_history**: Boolean to reset chat history

### Optional Inputs:
- **text_input**: Additional text input for context
- **image**: Image input (IMAGE type)
- **video**: Video frame sequence input (IMAGE type)
- **audio**: Audio input (AUDIO type)
- **max_output_tokens**: Set maximum output length (1-8192)
- **temperature**: Control response randomness (0.0-1.0)
- **structured_output**: Enable structured response format

## Usage Examples

### Basic Text Analysis:
```
Text Input Node -> Gemini Flash Node [input_type: "text"]
```

### Image Analysis:
```
Load Image Node -> Gemini Flash Node [input_type: "image"]
```

### Video Analysis:
```
Load Video Node -> Gemini Flash Node [input_type: "video"]
```

### Audio Analysis:
```
Load Audio Node -> Gemini Flash Node [input_type: "audio"]
```

## Chat Mode

Chat mode maintains conversation history and provides a more interactive experience:

1. Enable chat mode by setting `chat_mode: true`
2. Chat history format:
```
=== Chat History ===
USER: your message
ASSISTANT: Gemini's response
=== End History ===
```
3. Use `clear_history: true` to start a new conversation
4. Chat history persists between calls until cleared

### Chat Mode Tips:
- Works with all input types (text, image, video, audio)
- History is displayed in the output
- Maintains context across multiple interactions
- Clear history when switching topics

## Video Frame Handling

When processing videos:
- Automatically samples frames evenly throughout the video
- Resizes frames for efficient processing
- Works with both chat and non-chat modes

## Error Handling

The node provides clear error messages for common issues:
- Invalid API key
- Rate limit exceeded
- Invalid input formats
- Network/proxy issues

## Rate Limits

Default rate limits (from config.json):
- 10 requests per minute (RPM_LIMIT)
- 4 million tokens per minute (TPM_LIMIT)
- 1,500 requests per day (RPD_LIMIT)

## Contributing

Feel free to submit issues, fork the repository, and create pull requests for any improvements.

## License

MIT License

## Acknowledgments

- Google's Gemini API
- ComfyUI Community
- All contributors

---

**Note**: This node is experimental and based on Gemini 2.0 Flash Experimental model. Features and capabilities may change as the model evolves.
