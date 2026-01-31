import logging
import sys
import time
import functools
from pathlib import Path

# --- 1. CONFIGURATION ---
log_dir = Path("logs")
log_dir.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    handlers=[
        logging.FileHandler(log_dir / "app.log"),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger("CardioCDSS")

# --- 2. THE DECORATOR ---
def trace_task(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        func_name = func.__name__
        logger.info(f"🚀 Starting: {func_name}")
        start_time = time.perf_counter()
        
        try:
            result = func(*args, **kwargs)
            end_time = time.perf_counter()
            # Added a "Slow Task" alert logic here
            duration = end_time - start_time
            status = "✅ Finished" if duration < 10 else "⚠️ Finished (SLOW)"
            
            logger.info(f"{status}: {func_name} | Duration: {duration:.2f}s")
            return result
        except Exception as e:
            logger.error(f"❌ Failed: {func_name} | Error: {str(e)}", exc_info=True)
            raise e 
            
    return wrapper