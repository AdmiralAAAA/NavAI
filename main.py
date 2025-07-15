#!/usr/bin/env python3
"""
Lightweight Uzbek STT Bot for Railway
Optimized for deployment with minimal dependencies
"""

import os
import sys
import logging
import asyncio
import tempfile
from pathlib import Path
from typing import Optional
import json
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Try to import required packages with better error handling
def import_with_fallback(package_name, fallback_message=None):
    try:
        if package_name == "torch":
            import torch
            logger.info(f"✅ {package_name} imported successfully")
            return torch
        elif package_name == "torchaudio":
            import torchaudio
            logger.info(f"✅ {package_name} imported successfully")
            return torchaudio
        elif package_name == "transformers":
            from transformers import WhisperProcessor, WhisperForConditionalGeneration
            logger.info(f"✅ {package_name} imported successfully")
            return WhisperProcessor, WhisperForConditionalGeneration
        elif package_name == "telegram":
            from telegram import Update
            from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
            logger.info(f"✅ {package_name} imported successfully")
            return Update, Application, CommandHandler, MessageHandler, filters, ContextTypes
        elif package_name == "requests":
            import requests
            logger.info(f"✅ {package_name} imported successfully")
            return requests
    except ImportError as e:
        logger.error(f"❌ Failed to import {package_name}: {e}")
        if fallback_message:
            logger.info(fallback_message)
        return None

# Import all required packages
torch = import_with_fallback("torch")
torchaudio = import_with_fallback("torchaudio")
transformers_modules = import_with_fallback("transformers")
telegram_modules = import_with_fallback("telegram")
requests = import_with_fallback("requests")

# Check if all required modules are available
if not all([torch, torchaudio, transformers_modules, telegram_modules, requests]):
    logger.error("❌ Critical dependencies missing!")
    sys.exit(1)

# Unpack modules
WhisperProcessor, WhisperForConditionalGeneration = transformers_modules
Update, Application, CommandHandler, MessageHandler, filters, ContextTypes = telegram_modules

# Optional audio processing
try:
    import librosa
    import soundfile as sf
    AUDIO_LIBS = True
    logger.info("✅ Librosa and soundfile available")
except ImportError:
    AUDIO_LIBS = False
    logger.warning("⚠️ Audio processing libraries not available")

class UzbekSTTBot:
    def __init__(self, bot_token: str):
        self.bot_token = bot_token
        self.model_path = "islomov/navaistt_v2_medium"
        self.processor = None
        self.model = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Railway optimizations
        self.max_file_size = 10 * 1024 * 1024  # 10MB
        self.max_duration = 90  # 1.5 minutes
        self.model_loaded = False
        
        # Setup directories
        self.temp_dir = Path(tempfile.gettempdir()) / "uzbek_stt"
        self.temp_dir.mkdir(exist_ok=True)
        
        # Cache directory
        self.cache_dir = os.getenv('HF_HOME', '/tmp/huggingface')
        Path(self.cache_dir).mkdir(parents=True, exist_ok=True)
        
        logger.info(f"🔧 Device: {self.device}")
        logger.info(f"📁 Temp: {self.temp_dir}")
        logger.info(f"📦 Cache: {self.cache_dir}")
    
    async def load_model(self):
        """Load model with timeout and retries"""
        max_retries = 3
        for attempt in range(max_retries):
            try:
                logger.info(f"🔄 Loading model (attempt {attempt + 1}/{max_retries})...")
                
                # Load processor
                self.processor = WhisperProcessor.from_pretrained(
                    self.model_path,
                    cache_dir=self.cache_dir,
                    local_files_only=False
                )
                
                # Load model with CPU-only configuration
                self.model = WhisperForConditionalGeneration.from_pretrained(
                    self.model_path,
                    cache_dir=self.cache_dir,
                    torch_dtype=torch.float32,
                    low_cpu_mem_usage=True,
                    local_files_only=False
                )
                
                self.model.to(self.device)
                self.model.eval()
                
                self.model_loaded = True
                logger.info("✅ Model loaded successfully!")
                return True
                
            except Exception as e:
                logger.error(f"❌ Model loading attempt {attempt + 1} failed: {e}")
                if attempt < max_retries - 1:
                    logger.info(f"⏳ Retrying in 5 seconds...")
                    await asyncio.sleep(5)
                else:
                    logger.error("❌ All model loading attempts failed!")
                    return False
        
        return False
    
    async def download_audio(self, url: str, filepath: str) -> bool:
        """Download audio with progress tracking"""
        try:
            logger.info(f"📥 Downloading audio...")
            
            response = requests.get(url, stream=True, timeout=30)
            response.raise_for_status()
            
            total_size = 0
            with open(filepath, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        total_size += len(chunk)
                        if total_size > self.max_file_size:
                            raise ValueError(f"File too large: {total_size}")
                        f.write(chunk)
            
            logger.info(f"✅ Downloaded {total_size} bytes")
            return True
            
        except Exception as e:
            logger.error(f"❌ Download failed: {e}")
            return False
    
    def process_audio(self, audio_path: str) -> Optional[str]:
        """Process audio file to WAV format"""
        try:
            output_path = audio_path.replace(Path(audio_path).suffix, '.wav')
            
            if AUDIO_LIBS:
                # Use librosa for better audio processing
                audio, sr = librosa.load(audio_path, sr=16000, mono=True)
                sf.write(output_path, audio, 16000)
                logger.info("🎵 Audio processed with librosa")
            else:
                # Use torchaudio as fallback
                waveform, sample_rate = torchaudio.load(audio_path)
                
                # Resample if needed
                if sample_rate != 16000:
                    waveform = torchaudio.functional.resample(waveform, sample_rate, 16000)
                
                # Convert to mono
                if waveform.shape[0] > 1:
                    waveform = waveform.mean(dim=0, keepdim=True)
                
                # Save as WAV
                torchaudio.save(output_path, waveform, 16000)
                logger.info("🎵 Audio processed with torchaudio")
            
            return output_path
            
        except Exception as e:
            logger.error(f"❌ Audio processing failed: {e}")
            return None
    
    async def transcribe(self, audio_path: str) -> Optional[str]:
        """Transcribe audio to text"""
        try:
            if not self.model_loaded:
                logger.error("❌ Model not loaded!")
                return None
            
            # Process audio
            wav_path = self.process_audio(audio_path)
            if not wav_path:
                return None
            
            # Load processed audio
            waveform, sample_rate = torchaudio.load(wav_path)
            
            # Check duration
            duration = waveform.shape[1] / sample_rate
            if duration > self.max_duration:
                logger.warning(f"⚠️ Audio too long: {duration:.1f}s")
                return None
            
            logger.info(f"🎙️ Processing {duration:.1f}s audio")
            
            # Prepare input
            input_features = self.processor(
                waveform.squeeze().numpy(),
                sampling_rate=16000,
                return_tensors="pt"
            ).input_features.to(self.device)
            
            # Generate transcription
            with torch.no_grad():
                predicted_ids = self.model.generate(
                    input_features,
                    max_length=448,
                    num_beams=1,
                    do_sample=False
                )
            
            # Decode result
            transcription = self.processor.batch_decode(
                predicted_ids,
                skip_special_tokens=True
            )[0]
            
            # Cleanup
            if wav_path != audio_path and Path(wav_path).exists():
                Path(wav_path).unlink()
            
            logger.info(f"✅ Transcription completed: {len(transcription)} chars")
            return transcription.strip()
            
        except Exception as e:
            logger.error(f"❌ Transcription failed: {e}")
            return None
    
    def cleanup(self):
        """Clean up temporary files"""
        try:
            for file in self.temp_dir.glob("*"):
                if file.is_file():
                    file.unlink()
            logger.info("🧹 Cleanup completed")
        except Exception as e:
            logger.warning(f"⚠️ Cleanup warning: {e}")
    
    # Bot handlers
    async def start_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        status = "✅ Tayyor" if self.model_loaded else "❌ Model yuklanmoqda..."
        
        message = f"""
🎙️ **Uzbek STT Bot**

Salom! Audio xabarlaringizni matnga aylantiraman.

**Holat:** {status}

**Limitlar:**
• Max fayl: 10MB
• Max uzunlik: 90 soniya
• Formatlar: MP3, WAV, OGG

Audio yuboring! 🎵
        """
        await update.message.reply_text(message, parse_mode='Markdown')
    
    async def status_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Show bot status"""
        model_status = "✅ Yuklangan" if self.model_loaded else "❌ Yuklanmagan"
        device_info = f"📱 {self.device.upper()}"
        audio_support = "✅ To'liq" if AUDIO_LIBS else "⚠️ Asosiy"
        
        status_text = f"""
📊 **Bot holati:**

• Model: {model_status}
• Qurilma: {device_info}
• Audio: {audio_support}
• Temp fayllar: {len(list(self.temp_dir.glob('*')))}

**Limitlar:**
• Max fayl: {self.max_file_size // (1024*1024)}MB
• Max uzunlik: {self.max_duration}s
        """
        await update.message.reply_text(status_text, parse_mode='Markdown')
    
    async def handle_audio(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle audio messages"""
        if not self.model_loaded:
            await update.message.reply_text("⏳ Model hali yuklanmagan. Biroz kutib turing...")
            return
        
        try:
            # Clean up old files
            self.cleanup()
            
            msg = await update.message.reply_text("🔄 Qayta ishlanmoqda...")
            
            # Get audio file
            if update.message.voice:
                audio_file = update.message.voice
                ext = "ogg"
            elif update.message.audio:
                audio_file = update.message.audio
                ext = "mp3"
            else:
                await msg.edit_text("❌ Audio fayl topilmadi!")
                return
            
            # Check file size
            if audio_file.file_size > self.max_file_size:
                await msg.edit_text(f"❌ Fayl juda katta! Max: {self.max_file_size // (1024*1024)}MB")
                return
            
            # Check duration
            duration = getattr(audio_file, 'duration', 0)
            if duration > self.max_duration:
                await msg.edit_text(f"❌ Audio juda uzun! Max: {self.max_duration}s")
                return
            
            # Download file
            file_info = await context.bot.get_file(audio_file.file_id)
            file_path = self.temp_dir / f"audio_{int(time.time())}.{ext}"
            
            if not await self.download_audio(file_info.file_path, str(file_path)):
                await msg.edit_text("❌ Fayl yuklab olinmadi!")
                return
            
            # Transcribe
            await msg.edit_text("🎙️ Matn tanilmoqda...")
            result = await self.transcribe(str(file_path))
            
            # Clean up file
            if file_path.exists():
                file_path.unlink()
            
            if result:
                char_count = len(result)
                word_count = len(result.split())
                
                response = f"""
🎯 **Natija:**

{result}

---
📊 {char_count} belgi • {word_count} so'z • {duration}s
                """
                await msg.edit_text(response, parse_mode='Markdown')
            else:
                await msg.edit_text("❌ Matn tanilmadi. Qayta urinib ko'ring.")
                
        except Exception as e:
            logger.error(f"❌ Audio handling error: {e}")
            await update.message.reply_text("❌ Xatolik yuz berdi!")
    
    async def handle_text(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        await update.message.reply_text("🎙️ Menga audio xabar yuboring!\n\n/start - Boshiga qaytish\n/status - Bot holati")

async def main():
    """Main function"""
    logger.info("🚀 Starting Uzbek STT Bot...")
    
    # Get bot token
    bot_token = os.getenv('8116018261:AAFn6W-8uyWY_ibqjRBWFPQXmbt6DBcZNQs')
    if not bot_token:
        logger.error("❌ TELEGRAM_BOT_TOKEN not set!")
        sys.exit(1)
    
    # Create bot
    bot = UzbekSTTBot(bot_token)
    
    # Create application
    app = Application.builder().token(bot_token).build()
    
    # Add handlers
    app.add_handler(CommandHandler("start", bot.start_command))
    app.add_handler(CommandHandler("status", bot.status_command))
    app.add_handler(MessageHandler(filters.VOICE, bot.handle_audio))
    app.add_handler(MessageHandler(filters.AUDIO, bot.handle_audio))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, bot.handle_text))
    
    try:
        # Initialize
        await app.initialize()
        
        # Start bot
        await app.start()
        logger.info("✅ Bot application started")
        
        # Load model in background
        await bot.load_model()
        
        # Start polling
        await app.updater.start_polling(drop_pending_updates=True)
        logger.info("✅ Bot is running!")
        
        # Keep alive
        while True:
            await asyncio.sleep(1)
            
    except KeyboardInterrupt:
        logger.info("🛑 Bot stopped by user")
    except Exception as e:
        logger.error(f"❌ Fatal error: {e}")
    finally:
        bot.cleanup()
        await app.updater.stop()
        await app.stop()
        await app.shutdown()

if __name__ == "__main__":
    asyncio.run(main())
