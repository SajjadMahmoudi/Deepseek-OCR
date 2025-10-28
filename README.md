# DeepSeek OCR Telegram Bot

A powerful Telegram bot that extracts text from images and PDF documents using DeepSeek's advanced OCR model.

## Features

- 📷 **Image OCR**: Extract text from images with high accuracy
- 📄 **PDF Processing**: Extract text from all pages of PDF documents
- 🚀 **GPU Accelerated**: Utilizes CUDA for fast processing
- 📤 **Smart Output**: Automatically sends long results as downloadable text files
- 🔄 **Error Handling**: Built-in retry logic and robust error management

## Prerequisites

- Python 3.8 or higher
- NVIDIA GPU with CUDA support
- Telegram Bot Token ([Get one from @BotFather](https://t.me/botfather))

## Installation

1. **Clone the repository**
```bash
git clone <your-repo-url>
cd <repo-directory>
```

2. **Install required dependencies**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install transformers pillow pymupdf python-telegram-bot
```

3. **Configure the bot**

Open `test.py` and replace the `BOT_TOKEN` with your actual Telegram bot token:
```python
BOT_TOKEN = "your_bot_token_here"
```

## Usage

1. **Start the bot**
```bash
python test.py
```

2. **Interact with the bot on Telegram**
   - Send `/start` to see the welcome message
   - Send any image to extract text
   - Send a PDF document to extract text from all pages

## How It Works

### Image Processing
- Receives images via Telegram
- Processes using DeepSeek-OCR model with optimized settings
- Returns extracted text directly or as a file (for long outputs)

### PDF Processing
- Converts PDF pages to high-quality images (144 DPI)
- Processes each page individually with OCR
- Combines all pages into a single text output
- Returns results as a downloadable text file

## Model Information

This bot uses the [DeepSeek-OCR](https://huggingface.co/deepseek-ai/DeepSeek-OCR) model, which provides:
- High accuracy text extraction
- Support for various languages
- Efficient processing with GPU acceleration
- Advanced crop mode for better results

## Configuration

### Adjustable Parameters

- **DPI Setting**: Modify `dpi=144` in `pdf_to_images_high_quality()` for different quality levels
- **Base Size**: Change `base_size=1024` in `run_ocr()` for different processing sizes
- **Image Size**: Adjust `image_size=640` for optimal performance
- **Retry Attempts**: Modify `retries = 2` in `handle_image()` for error recovery

## Error Handling

The bot includes comprehensive error handling:
- Automatic retry on OCR failures
- Graceful handling of empty results
- Clear error messages for users
- Automatic cleanup of temporary files

## Output Format

- **Short text** (< 3800 characters): Sent as formatted message
- **Long text** (> 3800 characters): Sent as downloadable `.txt` file
- **PDF results**: Always sent as downloadable file with page separation

## Requirements

```
torch
transformers
python-telegram-bot
Pillow
PyMuPDF (fitz)
```

## Limitations

- Requires CUDA-capable GPU
- Model loads into GPU memory (~several GB)
- Processing time depends on image/PDF complexity
- First run downloads the DeepSeek-OCR model (~2-3 GB)

## Security Note

⚠️ **Important**: Never commit your bot token to public repositories. Consider using environment variables:

```python
import os
BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
```

## Troubleshooting

### Model fails to load
- Ensure you have sufficient GPU memory
- Check CUDA installation: `torch.cuda.is_available()`

### OCR returns empty results
- Verify image quality and contrast
- Ensure text is clearly visible in the image
- Try increasing DPI for PDF processing

### Bot doesn't respond
- Verify bot token is correct
- Check network connectivity
- Ensure the bot is running without errors

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is open source and available under the MIT License.

## Acknowledgments

- [DeepSeek AI](https://huggingface.co/deepseek-ai/DeepSeek-OCR) for the OCR model
- [python-telegram-bot](https://github.com/python-telegram-bot/python-telegram-bot) for the Telegram bot framework
- [PyMuPDF](https://pymupdf.readthedocs.io/) for PDF processing

## Support

For issues and questions, please open an issue on the GitHub repository.

---

**Built with ❤️ using DeepSeek-OCR**