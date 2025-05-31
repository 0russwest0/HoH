from dataclasses import dataclass, field, asdict
from typing import List, Dict, Tuple, Optional
from datetime import datetime

@dataclass
class Document:
    id: str
    title: str

@dataclass
class OutdatedInfo:
    answer: str
    evidence: str
    last_modified_time: str

@dataclass
class QA:
    question: str
    answer: str
    last_modified_time: str
    evidence: str
    document: Document
    outdated_infos: List[OutdatedInfo] = field(default_factory=list)

@dataclass
class Diff:
    id: str
    title: str
    text_old: str
    text_new: str
    diffs: List[Tuple[str, str]]  # List of (original_sentence, new_sentence) pairs

@dataclass
class ProcessingInput:
    question: Optional[str]
    old_answer: Optional[str]
    old_sentence: str
    new_sentence: str
    old_content: str
    new_content: str
    source_content: Optional[str] = None
    qa_info: Optional[Dict] = None
    document_info: Optional[Dict] = None

    def to_dict(self) -> Dict:
        """Convert the ProcessingInput object to a dictionary"""
        return {
            "question": self.question,
            "old_answer": self.old_answer,
            "old_sentence": self.old_sentence,
            "new_sentence": self.new_sentence,
            "old_content": self.old_content,
            "new_content": self.new_content,
            "source_content": self.source_content,
            "qa_info": self.qa_info,
            "document_info": self.document_info
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'ProcessingInput':
        """Create a ProcessingInput object from a dictionary"""
        return cls(**data)

@dataclass
class ProcessingResult:
    success: bool
    question: Optional[str] = None
    answer_old: Optional[str] = None
    answer_new: Optional[str] = None
    error: Optional[str] = None 

    def to_dict(self) -> Dict:
        """Convert result to dictionary for saving"""
        return {
            "success": self.success,
            "question": self.question,
            "answer_old": self.answer_old,
            "answer_new": self.answer_new,
            "error": self.error
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'ProcessingResult':
        """Create result from dictionary"""
        return cls(**data)

@dataclass
class ProcessingState:
    """Track the current state of processing"""
    step: str  # Current processing step (e.g., "update", "generation")
    processed_count: int  # Number of items processed
    total_count: int  # Total number of items to process
    
    def to_dict(self) -> Dict:
        """Convert state to dictionary for saving"""
        return {
            "step": self.step,
            "processed_count": self.processed_count,
            "total_count": self.total_count
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'ProcessingState':
        """Create state from dictionary"""
        return cls(
            step=data["step"],
            processed_count=data["processed_count"],
            total_count=data["total_count"]
        )

@dataclass
class QAResults:
    """Container for QA processing results"""
    updated_qas: List[Dict] = field(default_factory=list)  # Successfully updated QAs
    new_qas: List[Dict] = field(default_factory=list)  # Successfully generated new QAs
    failed_updates: List[ProcessingInput] = field(default_factory=list)  # Failed updates for retry 