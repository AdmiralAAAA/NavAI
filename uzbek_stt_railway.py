# ============================================================================
# UZBEK STT BOT - Railway.com Deployment Ready
# ============================================================================

import os
import logging
import asyncio
import torch
import torchaudio
from pathlib import Path
from typing import Optional
import tempfile
import sys
import gc

from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import requests

# Audio processing libraries
try:
    import librosa
    import soundfile as sf
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False
    print("Warning: librosa not available, audio conversion limited")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class UzbekSTTBot:
    def __init__(self, bot_token: str, model_path: str = "islomov/navaistt_v2_medium"):
        self.bot_token = bot_token
        self.model_path = model_path
        self.processor = None
        self.model = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.temp_dir = Path(tempfile.gettempdir()) / "uzbek_stt_temp"
        self.temp_dir.mkdir(exist_ok=True)
        self.librosa_available = LIBROSA_AVAILABLE
        
        # Railway-specific configurations
        self.max_file_size = 20 * 1024 * 1024  # 20MB for Railway
        self.max_duration = 180  # 3 minutes max for Railway
        
        logger.info(f"🔧 Device: {self.device}")
        logger.info(f"🎵 Audio conversion: {'✅' if self.librosa_available else '❌'}")
        logger.info(f"🎯 Model: {self.model_path}")
        logger.info(f"📁 Temp dir: {self.temp_dir}")
    
    def _convert_audio_to_wav(self, input_path: str, output_path: str) -> bool:
        """Convert audio to WAV using librosa or fallback methods"""
        try:
            if self.librosa_available:
                audio_data, _ = librosa.load(input_path, sr=16000, mono=True)
                sf.write(output_path, audio_data, 16000)
                return True
            else:
                # Fallback: try with torchaudio
                try:
                    waveform, sample_rate = torchaudio.load(input_path)
                    if sample_rate != 16000:
                        waveform = torchaudio.functional.resample(waveform, sample_rate, 16000)
                    if waveform.shape[0] > 1:
                        waveform = waveform.mean(dim=0, keepdim=True)
                    torchaudio.save(output_path, waveform, 16000)
                    return True
                except Exception as e:
                    logger.error(f"Torchaudio conversion failed: {e}")
                    return False
        except Exception as e:
            logger.error(f"Audio conversion failed: {e}")
            return False
    
    def _chunk_audio(self, waveform, chunk_length_seconds=25, overlap_seconds=3):
        """Split long audio into overlapping chunks (Railway-optimized)"""
        sample_rate = 16000
        chunk_samples = chunk_length_seconds * sample_rate
        overlap_samples = overlap_seconds * sample_rate
        
        chunks = []
        total_samples = waveform.shape[1]
        
        if total_samples <= chunk_samples:
            return [waveform]
        
        start = 0
        while start < total_samples:
            end = min(start + chunk_samples, total_samples)
            chunk = waveform[:, start:end]
            chunks.append(chunk)
            
            start += chunk_samples - overlap_samples
            
            if end >= total_samples:
                break
        
        return chunks
    
    async def load_model(self):
        """Load the model with Railway-specific optimizations"""
        try:
            logger.info(f"🔄 Loading model: {self.model_path}")
            
            # Set cache directory for Railway
            cache_dir = os.getenv('HF_HOME', '/tmp/huggingface')
            os.makedirs(cache_dir, exist_ok=True)
            
            # Load with reduced memory usage
            self.processor = WhisperProcessor.from_pretrained(
                self.model_path,
                cache_dir=cache_dir
            )
            
            self.model = WhisperForConditionalGeneration.from_pretrained(
                self.model_path,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                low_cpu_mem_usage=True,
                cache_dir=cache_dir
            )
            
            self.model.to(self.device)
            self.model.eval()
            
            if self.device == "cuda":
                self.model.half()
            
            logger.info("✅ Model loaded successfully!")
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    async def download_audio(self, file_url: str, file_path: str) -> bool:
        """Download audio file with timeout and size limits"""
        try:
            response = requests.get(file_url, timeout=30, stream=True)
            response.raise_for_status()
            
            downloaded_size = 0
            with open(file_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        downloaded_size += len(chunk)
                        if downloaded_size > self.max_file_size:
                            raise Exception(f"File too large: {downloaded_size} bytes")
                        f.write(chunk)
            
            return True
        except Exception as e:
            logger.error(f"Download failed: {e}")
            return False
    
    async def transcribe_audio_chunk(self, waveform_chunk) -> str:
        """Transcribe a single audio chunk with memory management"""
        try:
            # Process chunk
            input_features = self.processor(
                waveform_chunk.squeeze().numpy(),
                sampling_rate=16000,
                return_tensors="pt"
            ).input_features.to(self.device)
            
            if self.device == "cuda":
                input_features = input_features.half()
            
            # Generate transcription with memory optimization
            with torch.no_grad():
                predicted_ids = self.model.generate(
                    input_features,
                    max_length=448,
                    num_beams=1,
                    do_sample=False,
                    use_cache=True,
                    pad_token_id=self.processor.tokenizer.eos_token_id
                )
            
            # Decode
            transcription = self.processor.batch_decode(
                predicted_ids, 
                skip_special_tokens=True
            )[0]
            
            # Clean up GPU memory
            del input_features, predicted_ids
            if self.device == "cuda":
                torch.cuda.empty_cache()
            
            return transcription.strip()
            
        except Exception as e:
            logger.error(f"Chunk transcription failed: {e}")
            return ""
    
    async def transcribe_audio(self, audio_path: str) -> Optional[str]:
        """Transcribe audio with Railway-optimized processing"""
        try:
            logger.info(f"🎙️ Transcribing: {audio_path}")
            
            # Convert to WAV if needed
            wav_path = audio_path
            if not audio_path.lower().endswith('.wav'):
                wav_path = str(self.temp_dir / f"{Path(audio_path).stem}_converted.wav")
                if not self._convert_audio_to_wav(audio_path, wav_path):
                    logger.error("Failed to convert audio to WAV")
                    return None
            
            # Load audio with error handling
            try:
                waveform, sample_rate = torchaudio.load(wav_path)
            except Exception as e:
                logger.error(f"Failed to load audio: {e}")
                return None
            
            if sample_rate != 16000:
                waveform = torchaudio.functional.resample(waveform, sample_rate, 16000)
            
            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0, keepdim=True)
            
            # Calculate duration
            duration = waveform.shape[1] / 16000
            logger.info(f"⏱️ Audio duration: {duration:.1f} seconds")
            
            # Check duration limit
            if duration > self.max_duration:
                logger.warning(f"Audio too long: {duration}s > {self.max_duration}s")
                return None
            
            # Process based on length
            if duration <= 25:
                # Short audio - process directly
                logger.info("🔄 Processing short audio...")
                transcription = await self.transcribe_audio_chunk(waveform)
            else:
                # Long audio - process in chunks
                logger.info("🔄 Processing long audio in chunks...")
                chunks = self._chunk_audio(waveform, chunk_length_seconds=25, overlap_seconds=3)
                logger.info(f"📦 Split into {len(chunks)} chunks")
                
                transcriptions = []
                for i, chunk in enumerate(chunks):
                    logger.info(f"🔄 Processing chunk {i+1}/{len(chunks)}")
                    chunk_text = await self.transcribe_audio_chunk(chunk)
                    if chunk_text:
                        transcriptions.append(chunk_text)
                    
                    # Memory cleanup between chunks
                    gc.collect()
                    if self.device == "cuda":
                        torch.cuda.empty_cache()
                
                # Combine transcriptions
                transcription = " ".join(transcriptions)
                logger.info(f"🔗 Combined {len(transcriptions)} chunks")
            
            # Cleanup temporary files
            if wav_path != audio_path and Path(wav_path).exists():
                Path(wav_path).unlink()
            
            return transcription.strip() if transcription else None
            
        except Exception as e:
            logger.error(f"Transcription failed: {e}")
            return None
    
    async def start_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        message = """
🎙️ **Uzbek Speech-to-Text Bot**
🎯 **NavaiSTT v2 Model**

Salom! Uzbek tilida audio xabarlarni matnga aylantiraman.

**Xususiyatlari:**
• NavaiSTT v2 Medium model
• Audio qo'llab-quvvatlash (3 daqiqagacha)
• Yuqori aniqlik

**Buyruqlar:**
/start - Botni ishga tushirish
/stats - Bot ma'lumotlari
/help - Yordam

Audio xabar yuboring! 🎵
        """
        await update.message.reply_text(message, parse_mode='Markdown')
    
    async def stats_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        model_name = self.model_path.split('/')[-1]
        gpu_info = "✅ GPU" if self.device == "cuda" else "❌ CPU"
        
        stats = f"""
📊 **Bot statistikasi**

🚀 **Railway.com:**
• GPU: {gpu_info}
• Model: {model_name}
• Audio: {'✅ Librosa' if self.librosa_available else '⚠️ Torchaudio'}
• Holat: {"✅ Yuklangan" if self.model else "❌ Yuklanmagan"}

⚡ **Limits:**
• Max fayl: {self.max_file_size // (1024*1024)}MB
• Max uzunlik: {self.max_duration}s
• Chunk: 25s
        """
        await update.message.reply_text(stats, parse_mode='Markdown')
    
    async def help_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        help_text = """
❓ **Yordam**

**Qanday ishlatish:**
1. Audio xabar yuboring
2. Ovozli xabar yuboring
3. Audio fayl yuboring (.mp3, .wav, .ogg)

**Cheklovlar:**
• Fayl hajmi: 20MB gacha
• Uzunlik: 3 daqiqa gacha
• Formatlar: MP3, WAV, OGG

**Maslahatlar:**
• Aniq talaffuz qiling
• Shovqinni kamaytiring
• Uzbek tilida gapiring

Savollar? @your_username ga yozing.
        """
        await update.message.reply_text(help_text, parse_mode='Markdown')
    
    async def handle_audio(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle audio messages with Railway-optimized processing"""
        if not self.model:
            await update.message.reply_text("❌ Model yuklanmagan. Biroz kuting...")
            return
        
        try:
            msg = await update.message.reply_text("🔄 Qayta ishlanmoqda...")
            
            # Get audio info
            if update.message.voice:
                audio_file = update.message.voice
                file_ext = "ogg"
                duration = getattr(audio_file, 'duration', 0)
            elif update.message.audio:
                audio_file = update.message.audio
                file_ext = "mp3"
                duration = getattr(audio_file, 'duration', 0)
            else:
                await msg.edit_text("❌ Audio topilmadi.")
                return
            
            # Check limits
            if audio_file.file_size > self.max_file_size:
                await msg.edit_text(f"❌ Fayl juda katta (max {self.max_file_size // (1024*1024)}MB).")
                return
            
            if duration and duration > self.max_duration:
                await msg.edit_text(f"❌ Audio juda uzun (max {self.max_duration}s).")
                return
            
            # Show processing message
            if duration and duration > 30:
                await msg.edit_text(f"🔄 Uzun audio qayta ishlanmoqda ({duration}s)...\nBiroz sabr qiling.")
            
            # Download with unique filename
            file_info = await context.bot.get_file(audio_file.file_id)
            file_path = self.temp_dir / f"{audio_file.file_unique_id}.{file_ext}"
            
            if not await self.download_audio(file_info.file_path, str(file_path)):
                await msg.edit_text("❌ Fayl yuklab olinmadi.")
                return
            
            # Transcribe
            await msg.edit_text("🎙️ Matn tanilmoqda...")
            transcription = await self.transcribe_audio(str(file_path))
            
            # Cleanup
            if file_path.exists():
                file_path.unlink()
            
            if transcription:
                # Format response
                char_count = len(transcription)
                word_count = len(transcription.split())
                
                response = f"""
🎯 **Tanilgan matn:**

{transcription}

---
📊 {char_count} belgi, {word_count} so'z | ⏱️ {duration}s | 🎯 NavaiSTT v2
                """
                await msg.edit_text(response, parse_mode='Markdown')
            else:
                await msg.edit_text("❌ Matn tanib olinmadi. Boshqa audio sinab ko'ring.")
                
        except Exception as e:
            logger.error(f"Audio handling error: {e}")
            await update.message.reply_text("❌ Xatolik yuz berdi. Qayta urinib ko'ring.")
    
    async def handle_text(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        await update.message.reply_text("🎙️ Menga audio xabar yuboring. /help - batafsil ma'lumot")

# ============================================================================
# MAIN FUNCTION FOR RAILWAY
# ============================================================================

def main():
    """Main function for Railway deployment"""
    logger.info("🚀 Starting Uzbek STT Bot on Railway...")
    
    # Get token from environment
    bot_token = os.getenv('8116018261:AAFn6W-8uyWY_ibqjRBWFPQXmbt6DBcZNQs')
    if not bot_token:
        logger.error("❌ TELEGRAM_BOT_TOKEN not found in environment variables")
        sys.exit(1)
    
    # Set up environment
    os.environ.setdefault('HF_HOME', '/tmp/huggingface')
    
    # Create event loop
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    try:
        loop.run_until_complete(run_bot(bot_token))
    except KeyboardInterrupt:
        logger.info("🛑 Bot stopped by user")
    except Exception as e:
        logger.error(f"❌ Fatal error: {e}")
        sys.exit(1)
    finally:
        loop.close()

async def run_bot(bot_token: str):
    """Run the bot with proper error handling"""
    try:
        # Create bot
        bot = UzbekSTTBot(bot_token, "islomov/navaistt_v2_medium")
        
        # Create application
        app = Application.builder().token(bot_token).build()
        
        # Add handlers
        app.add_handler(CommandHandler("start", bot.start_command))
        app.add_handler(CommandHandler("stats", bot.stats_command))
        app.add_handler(CommandHandler("help", bot.help_command))
        
        # Audio handling
        app.add_handler(MessageHandler(filters.VOICE, bot.handle_audio))
        app.add_handler(MessageHandler(filters.AUDIO, bot.handle_audio))
        app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, bot.handle_text))
        
        # Initialize and load model
        await app.initialize()
        await bot.load_model()
        await app.start()
        
        logger.info("✅ Bot started successfully!")
        logger.info("🎯 Using NavaiSTT v2 model")
        logger.info("🚀 Ready to process audio messages")
        
        # Start polling
        await app.updater.start_polling(drop_pending_updates=True)
        
        # Keep running
        while True:
            await asyncio.sleep(1)
            
    except Exception as e:
        logger.error(f"❌ Bot error: {e}")
        raise
    finally:
        if 'app' in locals():
            await app.updater.stop()
            await app.stop()
            await app.shutdown()

if __name__ == "__main__":
    main()
