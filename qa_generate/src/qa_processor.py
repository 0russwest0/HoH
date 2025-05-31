import logging
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from tqdm import tqdm
from collections import defaultdict
import json

from .config import (
    BATCH_SIZE, INTERMEDIATE_DIR, DATA_DIR,
    OLD_MONTH, NEW_MONTH, INPUT_QA_FILE,OUTPUT_QA_FILE
)
from .models import ProcessingInput, ProcessingResult, ProcessingState
from .utils import (
    get_content, save_jsonl, load_jsonl, load_json,
    save_state, load_state, load_qa_results, clear_qa_results,
    append_updated_qa, append_new_qa, append_failed_update,
    merge_qa_files
)
from .llm_chains import LLMChainManager

class QAProcessor:
    def __init__(self):
        self.llm_manager = LLMChainManager()
        self.doc2qa = defaultdict(list)
        self.doc2diffs = {}
        self.qa_update = []
        self.qa_new = []
    
    def load_data(self, qa_file: Optional[Path], diff_file: Path) -> None:
        """Load and preprocess QA and diff data"""
        logging.info("Loading input data")
        
        # Load QAs from JSON file if it exists
        if qa_file is not None:
            qas = load_json(qa_file)
            for qa in tqdm(qas, desc="Processing QAs"):
                self.doc2qa[qa["document"]["id"]].append(qa)
            logging.info(f"Loaded {len(self.doc2qa)} documents with QAs")
        
        # Load diffs from JSONL file
        diffs = load_jsonl(diff_file)
        for diff in tqdm(diffs, desc="Processing diffs"):
            self.doc2diffs[diff["id"]] = diff
        logging.info(f"Loaded {len(self.doc2diffs)} documents with diffs")
    
    def classify_updates(self) -> None:
        """Classify QAs into updates and new generations"""
        logging.info("Classifying QA updates and new generations")
        
        for doc_id, diffs in tqdm(self.doc2diffs.items(), desc="Classifying QAs"):
            if doc_id not in self.doc2qa:
                self.qa_new.append(diffs)
                continue
            
            same_pairs = []
            new_diffs = []
            qa_list = self.doc2qa[doc_id]
            text_old = diffs["text_old"]
            text_new = diffs["text_new"]
            
            # Track used QAs
            used_qa_evidence = set()
            evidence_to_qa = {qa["evidence"]: qa for qa in qa_list}
            
            # Process each diff
            for diff in diffs["diffs"]:
                original_sentence = diff[0]
                if original_sentence in evidence_to_qa and original_sentence not in used_qa_evidence:
                    same_pairs.append((evidence_to_qa[original_sentence], diff, text_old, text_new))
                    used_qa_evidence.add(original_sentence)
                else:
                    new_diffs.append(diff)
            
            # Update results
            if same_pairs:
                self.qa_update.extend(same_pairs)
                diffs["diffs"] = []
            
            if new_diffs:
                diffs["diffs"] = new_diffs
                self.qa_new.append(diffs)
        
        logging.info(f"Classification complete. Updates: {len(self.qa_update)}, New: {len(self.qa_new)}")
    
    def prepare_update_inputs(self) -> List[ProcessingInput]:
        """Prepare inputs for QA updates"""
        inputs = []
        for qa, diff, text_old, text_new in self.qa_update:
            old_sentence, new_sentence = diff
            inputs.append(ProcessingInput(
                question=qa["question"],
                old_answer=qa["answer"],
                old_sentence=old_sentence,
                new_sentence=new_sentence,
                old_content=get_content(old_sentence, text_old),
                new_content=get_content(new_sentence, text_new),
                qa_info=qa
            ))
        return inputs
    
    def prepare_new_inputs(self) -> List[ProcessingInput]:
        """Prepare inputs for new QA generation"""
        inputs = []
        for diff in self.qa_new:
            title = diff["title"]
            old_passage = diff["text_old"]
            new_passage = diff["text_new"]
            
            for old_sentence, new_sentence in diff["diffs"]:
                old_content = get_content(old_sentence, old_passage)
                new_content = get_content(new_sentence, new_passage)
                source_content = (
                    f"## Title\n{title}\n"
                    f"## Old Content\n{old_content}\n"
                    f"## New Content\n{new_content}"
                )
                
                inputs.append(ProcessingInput(
                    question=None,
                    old_answer=None,
                    old_sentence=old_sentence,
                    new_sentence=new_sentence,
                    old_content=old_content,
                    new_content=new_content,
                    source_content=source_content,
                    document_info={"id": diff["id"], "title": diff["title"]}
                ))
        return inputs
    
    def prepare_failed_inputs(self, failed_updates: List[ProcessingInput]) -> List[ProcessingInput]:
        """Convert failed updates to new QA generation inputs"""
        failed_inputs = []
        for input_data in failed_updates:
            # Create source content for the new QA generation
            source_content = (
                f"## Title\n{input_data.qa_info['document']['title']}\n"
                f"## Old Content\n{input_data.old_content}\n"
                f"## New Content\n{input_data.new_content}"
            )
            
            # Convert to new QA generation input
            failed_inputs.append(ProcessingInput(
                question=None,  # Treat as new QA generation
                old_answer=None,
                old_sentence=input_data.old_sentence,
                new_sentence=input_data.new_sentence,
                old_content=input_data.old_content,
                new_content=input_data.new_content,
                source_content=source_content,
                document_info=input_data.qa_info["document"]
            ))
        
        logging.info(f"Prepared {len(failed_inputs)} inputs from failed updates")
        return failed_inputs
    
    def process_updates(self, inputs: List[ProcessingInput]) -> Tuple[List[dict], List[ProcessingInput]]:
        """Process QA updates with batch processing and incremental saving"""
        outputs = []
        failed_updates = []
        
        # Try to load existing state and results
        state = load_state(INTERMEDIATE_DIR)
        if state and state.step == "update":
            start_index = state.processed_count
            results = load_qa_results(INTERMEDIATE_DIR)
            outputs = results.updated_qas
            failed_updates = [ProcessingInput(**data) for data in results.failed_updates]
            assert len(outputs) + len(failed_updates) == start_index
            logging.info(f"Restored progress: {start_index} processed, {len(outputs)} successful, {len(failed_updates)} failed")
        else:
            start_index = 0
            state = ProcessingState(
                step="update",
                processed_count=0,
                total_count=len(inputs)
            )
            clear_qa_results(INTERMEDIATE_DIR, "update")  # Clear previous results
        
        # Process in batches
        for i in tqdm(range(start_index, len(inputs), BATCH_SIZE), desc="Updating QAs"):
            batch = inputs[i:i + BATCH_SIZE]
            batch_results = self.llm_manager.update_qa_batch(batch)
            
            # Handle results
            for input_data, result in zip(batch, batch_results):
                if result.success:
                    updated_qa = self.create_updated_qa(input_data.qa_info, result.answer_new, input_data.new_sentence)
                    outputs.append(updated_qa)
                    append_updated_qa(updated_qa, INTERMEDIATE_DIR)
                else:
                    failed_updates.append(input_data)
                    append_failed_update(input_data, INTERMEDIATE_DIR)
            
            # Update and save state
            state.processed_count = min(i + BATCH_SIZE, len(inputs))
            save_state(state, INTERMEDIATE_DIR)
            
            if (i + BATCH_SIZE) % 1000 == 0:  # Log progress every 1000 items
                logging.info(f"Processed {state.processed_count}/{len(inputs)} inputs. Success: {len(outputs)}, Failed: {len(failed_updates)}")
        
        logging.info(f"Processing complete. Updated QAs: {len(outputs)}, Failed updates: {len(failed_updates)}")
        return outputs, failed_updates
    
    def create_updated_qa(self, qa: Dict, new_answer: str, new_evidence: str) -> Dict:
        """Create updated QA entry"""
        updated_qa = qa.copy()
        updated_qa["answer"] = new_answer
        updated_qa["evidence"] = new_evidence
        updated_qa["outdated_infos"].append({
            "answer": qa["answer"],
            "evidence": qa["evidence"],
            "last_modified_time": qa["last_modified_time"]
        })
        updated_qa["last_modified_time"] = f"20{NEW_MONTH[0:2]}-{NEW_MONTH[2:]}-01"
        return updated_qa
    
    def create_new_qa(self, result: ProcessingResult, input_data: ProcessingInput) -> Dict:
        """Create new QA entry"""
        return {
            "question": result.question,
            "answer": result.answer_new,
            "last_modified_time": f"20{NEW_MONTH[0:2]}-{NEW_MONTH[2:]}-01",
            "evidence": input_data.new_sentence,
            "outdated_infos": [{
                "answer": result.answer_old,
                "evidence": input_data.old_sentence,
                "last_modified_time": f"20{OLD_MONTH[0:2]}-{OLD_MONTH[2:]}-01"
            }],
            "document": input_data.document_info
        }
    
    def process_qa_generation(self, inputs: List[ProcessingInput]) -> List[dict]:
        """Process QA generation with batch processing and incremental saving"""
        outputs = []
        
        # Try to load existing state and results
        state = load_state(INTERMEDIATE_DIR)
        if state and state.step == "generation":
            start_index = state.processed_count
            results = load_qa_results(INTERMEDIATE_DIR)
            outputs.extend(results.new_qas)
            logging.info(f"Restored progress: {start_index} processed, {len(results.new_qas)} successful QAs")
        else:
            start_index = 0
            state = ProcessingState(
                step="generation",
                processed_count=0,
                total_count=len(inputs)
            )
            clear_qa_results(INTERMEDIATE_DIR, "generation")  # Clear previous results
        
        # Process in batches
        for i in tqdm(range(start_index, len(inputs), BATCH_SIZE), desc="Generating QAs"):
            batch = inputs[i:i + BATCH_SIZE]
            batch_results = self.llm_manager.generate_qa_batch(batch)
            
            # Handle results
            for input_data, result in zip(batch, batch_results):
                if result.success:
                    new_qa = self.create_new_qa(result, input_data)
                    outputs.append(new_qa)
                    append_new_qa(new_qa, INTERMEDIATE_DIR)
            
            # Update and save state
            state.processed_count = min(i + BATCH_SIZE, len(inputs))
            save_state(state, INTERMEDIATE_DIR)
            
            if (i + BATCH_SIZE) % 1000 == 0:  # Log progress every 1000 items
                logging.info(f"Processed {state.processed_count}/{len(inputs)} inputs, Generated {len(outputs)} successful QAs")
        
        return outputs
    
    def needs_update(self) -> bool:
        """Check if QA updates are needed"""
        return len(self.doc2qa) > 0 
    
    def run(self, diff_file: Path, qa_file: Optional[Path] = None) -> None:
        """Run the complete QA processing pipeline
        
        Args:
            diff_file: Path to the diff file
            qa_file: Optional path to existing QA file. If None, only generate new QAs
        """
        # Load and preprocess data
        self.load_data(qa_file, diff_file)

        state = load_state(INTERMEDIATE_DIR)
        
        if self.needs_update():
            # Always classify updates first
            logging.info("Classifying QA updates and new generations")
            self.classify_updates()
            
            if state and state.step == "generation":
                results = load_qa_results(INTERMEDIATE_DIR)
                updated_qas = results.updated_qas
                failed_updates = [ProcessingInput(**data) for data in results.failed_updates]
                logging.info(f"Update processing already complete. Updated QAs: {len(updated_qas)}, Failed updates: {len(failed_updates)}")
                
                # Prepare all inputs for QA generation
                logging.info(f"Processing {len(failed_updates)} failed updates")
                generation_inputs = self.prepare_failed_inputs(failed_updates)
                
                logging.info("Preparing new QA inputs")
                new_inputs = self.prepare_new_inputs()
                logging.info(f"Found {len(new_inputs)} new inputs")
                generation_inputs.extend(new_inputs)
            else:
                # Process updates
                logging.info("Processing existing QAs for updates")
                update_inputs = self.prepare_update_inputs()
                updated_qas, failed_updates = self.process_updates(update_inputs)
                
                # Prepare all inputs for QA generation
                logging.info(f"Processing {len(failed_updates)} failed updates")
                generation_inputs = self.prepare_failed_inputs(failed_updates)
                
                logging.info("Preparing new QA inputs")
                new_inputs = self.prepare_new_inputs()
                logging.info(f"Found {len(new_inputs)} new inputs")
                generation_inputs.extend(new_inputs)
            
        else:
            # Initial generation without updates
            logging.info("No existing QAs found. Proceeding with initial QA generation")
            self.qa_new = list(self.doc2diffs.values())
            generation_inputs = self.prepare_new_inputs()
        
        logging.info(f"Total inputs for QA generation: {len(generation_inputs)}")
        
        # Process all QA generation inputs
        new_qas = self.process_qa_generation(generation_inputs)
        
        # Merge QA files
        if self.needs_update():
            logging.info(f"Processing complete. Updated QAs: {len(updated_qas)}, New QAs: {len(new_qas)}")
            mode = "update"
        else:
            logging.info(f"Initial QA generation complete. Generated QAs: {len(new_qas)}")
            mode = "init"
        
        # Merge QA files
        total_qas = merge_qa_files(
            old_qa_file=INPUT_QA_FILE if mode == "update" else None,
            updated_qa_file=INTERMEDIATE_DIR / "updated_qas.jsonl",
            new_qa_file=INTERMEDIATE_DIR / "new_qas.jsonl",
            output_file=OUTPUT_QA_FILE,
            mode=mode
        )
        logging.info(f"Merge complete. Total QAs: {total_qas}") 