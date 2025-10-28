import os
import torch
import fitz
import io
from io import BytesIO, StringIO
from pathlib import Path
from transformers import AutoModel, AutoTokenizer
from telegram import Update
from telegram.ext import ApplicationBuilder, MessageHandler, CommandHandler, filters, ContextTypes
from PIL import Image
import warnings
warnings.filterwarnings("ignore", message="The attention mask and the pad token id were not set")


#   Load model once  
model_name = "deepseek-ai/DeepSeek-OCR"
print("Loading OCR model...")
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModel.from_pretrained(model_name, trust_remote_code=True, use_safetensors=True)
model = model.eval().cuda().to(torch.bfloat16)
print("Model loaded successfully.")

#   Telegram bot token  
BOT_TOKEN = "Telegram_bot_token"  # replace with your actual bot token


#   PDF to Images conversion  
def pdf_to_images_high_quality(pdf_path, dpi=144):
    """Convert PDF pages to PIL images"""
    images = []
    pdf_document = fitz.open(pdf_path)
    
    zoom = dpi / 72.0
    matrix = fitz.Matrix(zoom, zoom)
    
    for page_num in range(pdf_document.page_count):
        page = pdf_document[page_num]
        pixmap = page.get_pixmap(matrix=matrix, alpha=False)
        Image.MAX_IMAGE_PIXELS = None
        
        img_data = pixmap.tobytes("png")
        img = Image.open(io.BytesIO(img_data))
        
        if img.mode in ('RGBA', 'LA'):
            background = Image.new('RGB', img.size, (255, 255, 255))
            background.paste(img, mask=img.split()[-1] if img.mode == 'RGBA' else None)
            img = background
        
        images.append(img)
    
    pdf_document.close()
    return images


#   OCR function for single image  
def run_ocr(image_path: str) -> str:
    import sys
    old_stdout = sys.stdout
    sys.stdout = captured_output = StringIO()

    try:
        model.infer(
            tokenizer,
            prompt="<image>\nFree OCR.",
            image_file=str(image_path),
            output_path=str(Path(image_path).parent),
            base_size=1024,
            image_size=640,
            crop_mode=True,
            save_results=False,
            test_compress=True
        )
    finally:
        sys.stdout = old_stdout

    captured_text = captured_output.getvalue()
    lines = captured_text.split('\n')
    ocr_lines = []

    start_index = 0
    for i, line in enumerate(lines):
        if line.strip() and not any(x in line for x in [
            'image size:', 'valid image', 'output texts', 'compression ratio:', '==============='
        ]):
            start_index = i
            break

    for i in range(start_index, len(lines)):
        line = lines[i].rstrip()
        if any(x in line for x in [
            'image size:', 'valid image tokens:', 'output texts tokens:', 'compression ratio:'
        ]):
            break
        if line.startswith('=================================================='):
            break
        if i < 4:
            continue
        ocr_lines.append(line)

    Final_text = '\n'.join(ocr_lines).strip()
    return Final_text or "(No text detected)"


#   OCR function for PIL Image object  
def run_ocr_pil(image: Image.Image, temp_path: str = "temp_ocr_image.jpg") -> str:
    """Run OCR on a PIL Image object"""
    image.save(temp_path)
    result = run_ocr(temp_path)
    if os.path.exists(temp_path):
        os.remove(temp_path)
    return result


#   Process PDF with OCR  
def process_pdf_ocr(pdf_path: str) -> str:
    """Extract text from all pages of a PDF"""
    images = pdf_to_images_high_quality(pdf_path)
    
    all_text = []
    for idx, image in enumerate(images, 1):
        page_text = run_ocr_pil(image, f"temp_page_{idx}.jpg")
        all_text.append(f"  Page {idx}  \n{page_text}")
    
    return "\n\n".join(all_text)


#   Handlers  
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "Send me an image or PDF and I'll extract text from it.\n\n"
        "📷 Images: Direct text extraction\n"
        "📄 PDFs: Text from all pages"
    )


async def handle_image(update: Update, context: ContextTypes.DEFAULT_TYPE):
    photo = update.message.photo[-1]
    file = await photo.get_file()
    image_path = Path("received_image.jpg")
    await file.download_to_drive(image_path)

    await update.message.reply_text("Processing image... please wait ⏳")

    try:
        retries = 2
        text = ""
        for attempt in range(retries):
            try:
                text = run_ocr(str(image_path))
                if text.strip():
                    break
            except Exception as e:
                print(f"[OCR ERROR] Attempt {attempt+1} failed: {e}")
                if attempt < retries - 1:
                    continue
                raise

        if not text.strip() or text.strip().lower() in ["no text detected", ""]:
            raise ValueError("OCR returned empty text")

        from telegram.helpers import escape_markdown
        safe_text = escape_markdown(text, version=2)

        if len(safe_text) > 3800:
            file = BytesIO(text.encode("utf-8"))
            file.name = "ocr_result.txt"
            await update.message.reply_document(file, caption="📄 Extracted OCR text")
        else:
            await update.message.reply_text(
                f"📜 *Extracted Text:*\n\n{safe_text}",
                parse_mode="MarkdownV2"
            )

    except Exception as e:
        print(f"[ERROR] OCR failed: {e}")
        await update.message.reply_text(
            "⚠️ OCR failed or no text detected. Please resend the image."
        )
    finally:
        if image_path.exists():
            os.remove(image_path)


async def handle_document(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle PDF documents"""
    document = update.message.document
    
    # Check if it's a PDF
    if not document.file_name.lower().endswith('.pdf'):
        await update.message.reply_text("⚠️ Please send a PDF file.")
        return
    
    file = await document.get_file()
    pdf_path = Path("received_document.pdf")
    await file.download_to_drive(pdf_path)
    
    await update.message.reply_text("📄 Processing PDF... this may take a moment ⏳")
    
    try:
        # Process the PDF
        text = process_pdf_ocr(str(pdf_path))
        
        if not text.strip() or text.strip().lower() in ["no text detected", ""]:
            raise ValueError("OCR returned empty text")
        
        # Always send PDF results as file (usually too long)
        file = BytesIO(text.encode("utf-8"))
        file.name = "pdf_ocr_result.txt"
        await update.message.reply_document(
            file, 
            caption=f"✅ Extracted text from PDF ({document.file_name})"
        )
        
    except Exception as e:
        print(f"[ERROR] PDF OCR failed: {e}")
        await update.message.reply_text(
            "⚠️ Failed to process PDF. Please ensure it contains readable images/text."
        )
    finally:
        if pdf_path.exists():
            os.remove(pdf_path)
        # Clean up any temp files
        for temp_file in Path(".").glob("temp_page_*.jpg"):
            temp_file.unlink()


#   Run bot  
def main():
    app = ApplicationBuilder().token(BOT_TOKEN).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.PHOTO, handle_image))
    app.add_handler(MessageHandler(filters.Document.PDF, handle_document))

    print("Bot is running...")
    app.run_polling()


if __name__ == "__main__":
    main()