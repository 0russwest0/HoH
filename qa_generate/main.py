import logging
from src.utils import setup_logging
from src.qa_processor import QAProcessor
from src.config import DIFF_FILE, INPUT_QA_FILE, MODE

def main():
    # Setup logging
    setup_logging()
    
    try:
        # Initialize processor
        processor = QAProcessor()
        
        # Run processing pipeline   
        if MODE == "init":
            processor.run(DIFF_FILE)
        elif MODE == "update":
            processor.run(DIFF_FILE, INPUT_QA_FILE)
        
    except Exception as e:
        logging.error(f"Error during processing: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    main() 