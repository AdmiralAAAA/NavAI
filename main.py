#!/usr/bin/env python3
"""
Basic Uzbek STT Bot - Railway Compatible
This version will start first, then we can add ML dependencies
"""

import os
import sys
import logging
import asyncio
import tempfile
from pathlib import Path
import json
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# First, let's test with just basic dependencies
try:
    import requests
    logger.info("✅ Requests imported successfully")
except ImportError as e:
    logger.error(f"❌ Failed to import requests: {e}")
    sys.exit(1)

# Try to import Telegram bot
try:
    from telegram import Update
    from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
    logger.info("✅ Telegram bot library imported successfully")
except ImportError as e:
    logger.error(f"❌ Failed to import telegram: {e}")
    sys.exit(1)

# Try ML libraries with graceful fallback
ML_AVAILABLE = False
try:
    import torch
    import torchaudio
    from transformers import WhisperProcessor, WhisperForConditionalGeneration
    ML_AVAILABLE = True
    logger.info("✅ ML libraries available")
except ImportError as e:
    logger.warning(f"⚠️ ML libraries not available: {e}")
    logger.info("🔄 Bot will run in basic mode")

# Audio processing libraries
AUDIO_AVAILABLE = False
try:
    import librosa
    import soundfile as sf
    AUDIO_AVAILABLE = True
    logger.info("✅ Audio processing libraries available")
except ImportError:
    logger.warning("⚠️ Audio processing libraries not available")

class BasicUzbekSTTBot:
    def __init__(self, bot_token: str):
        self.bot_token = bot_token
        self.ml_available = ML_AVAILABLE
        self.audio_available = AUDIO_AVAILABLE
        
        # ML components (only if available)
        self.processor = None
        self.model = None
        self.device = "cpu"
        self.model_loaded = False
        
        if ML_AVAILABLE:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.model_path = "islomov/navaistt_v2_medium"
        
        # Setup directories
        self.temp_dir = Path(tempfile.gettempdir()) / "uzbek_stt"
        self.temp_dir.mkdir(exist_ok=True)
        
        logger.info(f"🤖 Bot initialized")
        logger.info(f"🧠 ML: {'✅' if self.ml_available else '❌'}")
        logger.info(f"🎵 Audio: {'✅' if self.audio_available else '❌'}")
        logger.info(f"🔧 Device: {self.device if ML_AVAILABLE else 'N/A'}")
    
    async def load_model(self):
        """Load ML model if available"""
        if not self.ml_available:
            logger.info("⚠️ ML libraries not available, skipping model loading")
            return True
        
        try:
            logger.info("🔄 Loading NavaiSTT model...")
            
            cache_dir = os.getenv('HF_HOME', '/tmp/huggingface')
            os.makedirs(cache_dir, exist_ok=True)
            
            self.processor = WhisperProcessor.from_pretrained(
                self.model_path,
                cache_dir=cache_dir
            )
            
            self.model = WhisperForConditionalGeneration.from_pretrained(
                self.model_path,
                cache_dir=cache_dir,
                torch_dtype=torch.float32
            )
            
            self.model.to(self.device)
            self.model.eval()
            
            self.model_loaded = True
            logger.info("✅ Model loaded successfully!")
            return True
            
        except Exception as e:
            logger.error(f"❌ Model loading failed: {e}")
            return False
    
    async def download_audio(self, url: str, filepath: str) -> bool:
        """Download audio file"""
        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            
            with open(filepath, 'wb') as f:
                f.write(response.content)
            
            logger.info(f"📥 Downloaded audio: {len(response.content)} bytes")
            return True
            
        except Exception as e:
            logger.error(f"❌ Download failed: {e}")
            return False
    
    async def transcribe_audio(self, audio_path: str) -> str:
        """Transcribe audio (or return mock response if ML not available)"""
        if not self.ml_available or not self.model_loaded:
            return "⚠️ ML modellar yuklanmagan. Bu demo rejimda ishlamoqda."
        
        try:
            # Load and process audio
            waveform, sample_rate = torchaudio.load(audio_path)
            
            if sample_rate != 16000:
                waveform = torchaudio.functional.resample(waveform, sample_rate, 16000)
            
            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0, keepdim=True)
            
            # Process with model
            input_features = self.processor(
                waveform.squeeze().numpy(),
                sampling_rate=16000,
                return_tensors="pt"
            ).input_features.to(self.device)
            
            with torch.no_grad():
                predicted_ids = self.model.generate(
                    input_features,
                    max_length=448,
                    num_beams=1
                )
            
            transcription = self.processor.batch_decode(
                predicted_ids,
                skip_special_tokens=True
            )[0]
            
            return transcription.strip()
            
        except Exception as e:
            logger.error(f"❌ Transcription failed: {e}")
            return f"❌ Xatolik: {str(e)}"
    
    def cleanup(self):
        """Clean up temporary files"""
        try:
            for file in self.temp_dir.glob("*"):
                if file.is_file():
                    file.unlink()
        except Exception:
            pass
    
    # Bot command handlers
    async def start_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        ml_status = "✅ Tayyor" if (self.ml_available and self.model_loaded) else "❌ Mavjud emas"
        
        message = f"""
🎙️ **Uzbek STT Bot**

Salom! Audio xabarlaringizni matnga aylantiraman.

**Holat:**
• ML Model: {ml_status}
• Audio: {'✅' if self.audio_available else '❌'}
• Bot: ✅ Ishlayapti

Audio xabar yuboring! 🎵

/status - Batafsil holat
/test - Test xabari
        """
        await update.message.reply_text(message, parse_mode='Markdown')
    
    async def status_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Show detailed status"""
        status_text = f"""
📊 **Bot holati:**

**Kutubxonalar:**
• Requests: ✅
• Telegram: ✅
• PyTorch: {'✅' if ML_AVAILABLE else '❌'}
• Transformers: {'✅' if ML_AVAILABLE else '❌'}
• Librosa: {'✅' if AUDIO_AVAILABLE else '❌'}

**Model:**
• Yuklanishi: {'✅' if self.model_loaded else '❌'}
• Qurilma: {self.device if ML_AVAILABLE else 'N/A'}

**Fayllar:**
• Temp: {len(list(self.temp_dir.glob('*')))} fayl
• Katalog: {self.temp_dir}
        """
        await update.message.reply_text(status_text, parse_mode='Markdown')
    
    async def test_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Test command"""
        test_info = f"""
🧪 **Test natijasi:**

• Bot ishlayapti: ✅
• Vaqt: {time.strftime('%Y-%m-%d %H:%M:%S')}
• Python: {sys.version.split()[0]}
• Platform: Railway.app

Bu xabar bot to'g'ri ishlayotganini ko'rsatadi.
        """
        await update.message.reply_text(test_info, parse_mode='Markdown')
    
    async def handle_audio(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle audio messages"""
        try:
            msg = await update.message.reply_text("🔄 Audio qayta ishlanmoqda...")
            
            # Get audio file info
            if update.message.voice:
                audio_file = update.message.voice
                ext = "ogg"
            elif update.message.audio:
                audio_file = update.message.audio
                ext = "mp3"
            else:
                await msg.edit_text("❌ Audio fayl topilmadi!")
                return
            
            # Check if we can process
            if not self.ml_available:
                await msg.edit_text("""
❌ **ML kutubxonalar yuklanmagan!**

Bot hozirda demo rejimda ishlayapti.
Model yuklanishi uchun:
1. Deployment loglarini tekshiring
2. PyTorch va Transformers o'rnatilganini tasdiqlang
3. Railway resurslarini tekshiring

/status - Batafsil ma'lumot
                """, parse_mode='Markdown')
                return
            
            # Download and process
            file_info = await context.bot.get_file(audio_file.file_id)
            file_path = self.temp_dir / f"audio_{int(time.time())}.{ext}"
            
            if not await self.download_audio(file_info.file_path, str(file_path)):
                await msg.edit_text("❌ Fayl yuklab olinmadi!")
                return
            
            # Transcribe
            await msg.edit_text("🎙️ Matn tanilmoqda...")
            result = await self.transcribe_audio(str(file_path))
            
            # Clean up
            if file_path.exists():
                file_path.unlink()
            
            # Send result
            duration = getattr(audio_file, 'duration', 0)
            response = f"""
🎯 **Natija:**

{result}

---
📊 {len(result)} belgi • {duration}s
            """
            await msg.edit_text(response, parse_mode='Markdown')
            
        except Exception as e:
            logger.error(f"❌ Audio handling error: {e}")
            await update.message.reply_text(f"❌ Xatolik: {str(e)}")
    
    async def handle_text(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        await update.message.reply_text("""
🎙️ **Audio xabar yuboring!**

**Buyruqlar:**
/start - Boshiga qaytish
/status - Bot holati
/test - Test
        """, parse_mode='Markdown')

async def main():
    """Main function"""
    logger.info("🚀 Starting Basic Uzbek STT Bot...")
    
    # Get bot token
    bot_token = os.getenv('TELEGRAM_BOT_TOKEN')
    if not bot_token:
        logger.error("❌ TELEGRAM_BOT_TOKEN not set!")
        sys.exit(1)
    
    logger.info("✅ Bot token found")
    
    # Create bot
    bot = BasicUzbekSTTBot(bot_token)
    
    # Create application
    app = Application.builder().token(bot_token).build()
    
    # Add handlers
    app.add_handler(CommandHandler("start", bot.start_command))
    app.add_handler(CommandHandler("status", bot.status_command))
    app.add_handler(CommandHandler("test", bot.test_command))
    app.add_handler(MessageHandler(filters.VOICE, bot.handle_audio))
    app.add_handler(MessageHandler(filters.AUDIO, bot.handle_audio))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, bot.handle_text))
    
    try:
        # Initialize
        await app.initialize()
        logger.info("✅ Application initialized")
        
        # Start bot
        await app.start()
        logger.info("✅ Bot started")
        
        # Try to load model (if available)
        if ML_AVAILABLE:
            logger.info("🔄 Attempting to load ML model...")
            await bot.load_model()
        
        # Start polling
        await app.updater.start_polling(drop_pending_updates=True)
        logger.info("✅ Bot is running and polling for messages!")
        
        # Keep alive
        while True:
            await asyncio.sleep(10)
            logger.info("💓 Bot heartbeat - still running")
            
    except KeyboardInterrupt:
        logger.info("🛑 Bot stopped by user")
    except Exception as e:
        logger.error(f"❌ Fatal error: {e}")
        raise
    finally:
        logger.info("🧹 Cleaning up...")
        bot.cleanup()
        await app.updater.stop()
        await app.stop()
        await app.shutdown()

if __name__ == "__main__":
    logger.info("🎯 Python script starting...")
    asyncio.run(main())
