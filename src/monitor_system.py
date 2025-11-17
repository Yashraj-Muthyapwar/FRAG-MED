#!/usr/bin/env python3
"""
FRAG-MED System Monitoring Script
Launches Phoenix observability server and provides real-time monitoring interface
"""
import sys
import time
import signal
from pathlib import Path
import webbrowser
import logging
import coloredlogs
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import config
from src.observability import PhoenixObservability


class SystemMonitor:
    """System monitoring and Phoenix management"""
    
    def __init__(self):
        self.phoenix = None
        self.running = False
        
    def setup_logging(self):
        """Configure logging for monitoring"""
        log_file = config.LOGS_DIR / f"monitor_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler(sys.stdout)
            ]
        )
        
        # Add colored output
        coloredlogs.install(
            level='INFO',
            fmt='%(asctime)s - %(levelname)s - %(message)s'
        )
        
        return logging.getLogger(__name__)
    
    def start_phoenix(self):
        """Start Phoenix observability server"""
        logger = logging.getLogger(__name__)
        
        logger.info("="*80)
        logger.info("🔍 FRAG-MED SYSTEM MONITOR")
        logger.info("="*80)
        logger.info(f"\nProject: {config.PROJECT_ROOT}")
        logger.info(f"Data Directory: {config.DATA_DIR}")
        logger.info(f"ChromaDB: {config.CHROMADB_DIR}")
        logger.info(f"Parent Docs: {config.PARENT_DOCS_DIR}")
        logger.info("\nStarting Phoenix observability server...")
        
        try:
            # Initialize Phoenix
            self.phoenix = PhoenixObservability(
                project_name="frag-med-central",
                phoenix_host=config.PHOENIX_HOST,
                phoenix_port=config.PHOENIX_PORT,
                enable_tracing=True
            )
            
            dashboard_url = self.phoenix.get_dashboard_url()
            
            logger.info(f"\n✅ Phoenix server running!")
            logger.info(f"📊 Dashboard: {dashboard_url}")
            
            self.print_instructions(dashboard_url)
            
            # Auto-open browser
            try:
                logger.info("\n🌐 Opening browser...")
                webbrowser.open(dashboard_url)
                logger.info("✓ Browser opened automatically")
            except Exception as e:
                logger.warning(f"⚠️  Could not auto-open browser: {e}")
                logger.info(f"Please manually open: {dashboard_url}")
            
            self.running = True
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to start Phoenix: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def print_instructions(self, dashboard_url):
        """Print usage instructions"""
        print("\n" + "="*80)
        print("📊 PHOENIX DASHBOARD INSTRUCTIONS")
        print("="*80)
        print(f"\n1. Dashboard URL: {dashboard_url}")
        print("\n2. What you'll see:")
        print("   • Traces: Real-time query execution traces")
        print("   • Analytics: Latency, token usage, retrieval metrics")
        print("   • Evaluations: Response quality analysis")
        print("\n3. Run queries in another terminal:")
        print("   $ python test_queries.py")
        print("   OR")
        print("   $ python custom_query.py")
        print("\n4. Watch traces appear in real-time in the browser")
        print("\n5. Press Ctrl+C in this terminal to stop monitoring")
        print("="*80)
    
    def monitor_loop(self):
        """Main monitoring loop"""
        logger = logging.getLogger(__name__)
        
        logger.info("\n⏳ Monitoring active... (Press Ctrl+C to stop)\n")
        
        start_time = time.time()
        check_interval = 30  # Check every 30 seconds
        
        try:
            while self.running:
                # Sleep and allow interruption
                time.sleep(check_interval)
                
                # Calculate uptime
                uptime = time.time() - start_time
                uptime_mins = int(uptime / 60)
                uptime_secs = int(uptime % 60)
                
                # Print status update
                logger.info(
                    f"📊 Phoenix running | "
                    f"Uptime: {uptime_mins}m {uptime_secs}s | "
                    f"Dashboard: {self.phoenix.get_dashboard_url()}"
                )
                
        except KeyboardInterrupt:
            logger.info("\n\n🛑 Received shutdown signal...")
            self.shutdown()
    
    def shutdown(self):
        """Shutdown Phoenix and cleanup"""
        logger = logging.getLogger(__name__)
        
        logger.info("Stopping Phoenix server...")
        self.running = False
        
        if self.phoenix:
            self.phoenix.shutdown()
        
        logger.info("✅ Monitoring stopped")
        logger.info("\n" + "="*80)
        logger.info("Thank you for using FRAG-MED System Monitor!")
        logger.info("="*80 + "\n")


def signal_handler(signum, frame):
    """Handle interrupt signals gracefully"""
    print("\n\n🛑 Interrupt received. Shutting down...")
    sys.exit(0)


def main():
    """Main entry point"""
    
    # Register signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Create monitor
    monitor = SystemMonitor()
    
    # Setup logging
    logger = monitor.setup_logging()
    
    # Check if Phoenix is enabled
    if not config.ENABLE_PHOENIX:
        logger.error("Phoenix is disabled in config.py")
        logger.error("Set ENABLE_PHOENIX = True to use this monitor")
        sys.exit(1)
    
    # Start Phoenix server
    if monitor.start_phoenix():
        # Run monitoring loop
        try:
            monitor.monitor_loop()
        except Exception as e:
            logger.error(f"Error in monitoring loop: {e}")
            import traceback
            traceback.print_exc()
        finally:
            monitor.shutdown()
    else:
        logger.error("Failed to start monitoring system")
        sys.exit(1)


if __name__ == "__main__":
    main()
