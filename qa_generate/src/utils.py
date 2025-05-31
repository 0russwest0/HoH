import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
from .models import ProcessingResult, ProcessingInput, ProcessingState, QAResults

def setup_logging():
    """Configure logging settings"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Set specific loggers to WARNING
    for logger_name in ["openai", "httpx", "langchain"]:
        logging.getLogger(logger_name).setLevel(logging.WARNING)

def get_content(sentence: str, passage: str, max_words: int = 500) -> str:
    """Extract context around a sentence from a passage"""
    paragraphs = passage.split("\n")
    text = ""
    
    # Find the paragraph containing the sentence
    for i, paragraph in enumerate(paragraphs):
        if sentence in paragraph:
            text = paragraph
            break
    
    if not text:
        return ""
    
    # Truncate if too long
    words = text.split()
    if len(words) > max_words:
        return " ".join(words[:max_words])
    
    # Add surrounding context if needed
    j = 1
    while len(text.split()) < max_words and (i-j >= 0 or i+j < len(paragraphs)):
        if i-j >= 0:
            text_with_prev = paragraphs[i-j] + " " + text
            if len(text_with_prev.split()) <= max_words:
                text = text_with_prev
        
        if i+j < len(paragraphs):
            text_with_next = text + " " + paragraphs[i+j]
            if len(text_with_next.split()) <= max_words:
                text = text_with_next
        
        j += 1
    
    return text

def load_json(filepath: Path) -> Any:
    """Load data from a JSON file"""
    with open(filepath) as f:
        return json.load(f)

def save_jsonl(items: List[Dict[str, Any]], filepath: Path) -> None:
    """Save items to a JSONL file"""
    with open(filepath, 'w') as f:
        for item in items:
            f.write(json.dumps(item) + '\n')

def load_jsonl(filepath: Path) -> List[Dict[str, Any]]:
    """Load items from a JSONL file"""
    items = []
    with open(filepath) as f:
        for line in f:
            items.append(json.loads(line))
    return items

def save_state(state: 'ProcessingState', output_dir: Path) -> None:
    """Save processing state"""
    output_dir.mkdir(parents=True, exist_ok=True)
    state_path = output_dir / "state.json"
    
    with open(state_path, 'w') as f:
        json.dump(state.to_dict(), f)

def load_state(output_dir: Path) -> Optional['ProcessingState']:
    """Load processing state"""
    state_path = output_dir / "state.json"
    try:
        with open(state_path) as f:
            data = json.load(f)
            state = ProcessingState.from_dict(data)
            logging.info(f"Restored state: {state.step} - {state.processed_count}/{state.total_count}")
            return state
    except FileNotFoundError:
        return None

def append_updated_qa(result: dict, output_dir: Path) -> None:
    """Append a single updated QA result to file"""
    output_dir.mkdir(parents=True, exist_ok=True)
    qa_path = output_dir / "updated_qas.jsonl"
    
    with open(qa_path, 'a') as f:
        f.write(json.dumps(result) + '\n')

def append_new_qa(result: dict, output_dir: Path) -> None:
    """Append a single new QA result to file"""
    output_dir.mkdir(parents=True, exist_ok=True)
    qa_path = output_dir / "new_qas.jsonl"
    
    with open(qa_path, 'a') as f:
        f.write(json.dumps(result) + '\n')

def append_failed_update(update: 'ProcessingInput', output_dir: Path) -> None:
    """Append a single failed update to file"""
    output_dir.mkdir(parents=True, exist_ok=True)
    failed_path = output_dir / "failed_updates.jsonl"
    
    with open(failed_path, 'a') as f:
        f.write(json.dumps(update.to_dict()) + '\n')

def load_qa_results(output_dir: Path) -> 'QAResults':
    """Load all QA results from files"""
    results = QAResults()
    
    # Load updated QAs
    updated_path = output_dir / "updated_qas.jsonl"
    if updated_path.exists():
        with open(updated_path) as f:
            results.updated_qas = [json.loads(line) for line in f]
    
    # Load new QAs
    new_path = output_dir / "new_qas.jsonl"
    if new_path.exists():
        with open(new_path) as f:
            results.new_qas = [json.loads(line) for line in f]
    
    # Load failed updates
    failed_path = output_dir / "failed_updates.jsonl"
    if failed_path.exists():
        with open(failed_path) as f:
            results.failed_updates = [json.loads(line) for line in f]
    
    return results

def clear_qa_results(output_dir: Path, step: str) -> None:
    """Clear all QA results files"""
    if step == "update":
        for filename in ["updated_qas.jsonl", "failed_updates.jsonl"]:
            path = output_dir / filename
            if path.exists():
                path.unlink()
    elif step == "generation":
        for filename in ["new_qas.jsonl"]:
            path = output_dir / filename
            if path.exists():
                path.unlink()
    logging.info("Cleared all QA results files") 

def merge_qa_files(old_qa_file: Optional[Path], updated_qa_file: Path, new_qa_file: Path, output_file: Path, mode: str = "update") -> int:
    """Merge QA files based on mode
    
    Args:
        old_qa_file: Path to the old QA file (optional in init mode)
        updated_qa_file: Path to the updated QA file
        new_qa_file: Path to the new QA file
        output_file: Path to save the merged QAs
        mode: Processing mode ("init" or "update")
    
    Returns:
        int: Total number of QAs after merge
    """
    logging.info("Starting QA files merge")
    merged_qas = []
    
    if mode == "update":
        # Load old QAs
        with open(old_qa_file) as f:
            old_qas = json.load(f)
        logging.info(f"Loaded {len(old_qas)} old QAs")
        
        # Load updated QAs
        updated_qas = []
        if updated_qa_file.exists():
            with open(updated_qa_file) as f:
                updated_qas = [json.loads(line) for line in f]
            logging.info(f"Loaded {len(updated_qas)} updated QAs")
        
        # Merge old and updated QAs
        update_count = 0
        
        for qa in old_qas:
            updated = False
            for updated_qa in updated_qas:
                if (qa["question"] == updated_qa["question"] and 
                    qa["document"]["id"] == updated_qa["document"]["id"] and 
                    qa["evidence"] == updated_qa["outdated_infos"][-1]["evidence"]):
                    merged_qas.append(updated_qa)
                    updated = True
                    update_count += 1
                    break
            if not updated:
                merged_qas.append(qa)
        
        logging.info(f"Updated {update_count} QAs")
        
        # Add new QAs
        if new_qa_file.exists():
            with open(new_qa_file) as f:
                for line in f:
                    merged_qas.append(json.loads(line))
            logging.info(f"Added {len(merged_qas) - len(old_qas)} new QAs")
    
    else:  # init mode
        # In init mode, we only have new QAs
        if new_qa_file.exists():
            with open(new_qa_file) as f:
                for line in f:
                    merged_qas.append(json.loads(line))
            logging.info(f"Loaded {len(merged_qas)} new QAs")
    
    # Sort by document ID
    merged_qas = sorted(merged_qas, key=lambda x: x["document"]["id"])
    logging.info(f"Total QAs after merge: {len(merged_qas)}")
    
    # Save merged QAs
    with open(output_file, "w") as f:
        json.dump(merged_qas, f, indent=2)
    logging.info(f"Saved merged QAs to {output_file}")
    
    return len(merged_qas) 