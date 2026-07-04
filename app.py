import sys
import logging
from cli import cli

# Setup basic logging for the shim
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(name)s: %(message)s')
logger = logging.getLogger("LegacyShim")

if __name__ == '__main__':
    logger.warning("="*60)
    logger.warning("WARNING: You ran the legacy 'app.py' script.")
    logger.warning("Pi Vision AI has been successfully upgraded to the v1.0 architecture.")
    logger.warning("Automatically forwarding your request to the new execution engine...")
    logger.warning("In the future, you can also run: python cli.py run")
    logger.warning("="*60)
    
    # We simulate running `python cli.py run`
    try:
        # Pass 'run' as the default command if no arguments are provided
        args = sys.argv[1:] if len(sys.argv) > 1 else ['run']
        cli(args)
    except Exception as e:
        logger.error(f"Failed to start v1.0 engine: {e}")
