from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnableLambda
import re
from typing import Tuple, Optional, List, Dict
from .config import MODEL_CONFIG
from .models import ProcessingResult, ProcessingInput
from .templates import (
    QA_GENERATE_TEMPLATES,
    QA_TEMPLATE_WITH_CONTEXT,
    QA_CHECK_SYSTEM_TEMPLATES,
    QA_CHECK_USER_TEMPLATES,
    SAME_ANSWER_CHECK_TEMPLATE
)
import logging

class LLMChainManager:
    def __init__(self):
        self.model = ChatOpenAI(
            model=MODEL_CONFIG["name"],
            api_key=MODEL_CONFIG["api_key"],
            base_url=MODEL_CONFIG["base_url"],
            temperature=MODEL_CONFIG["temperature"],
            max_tokens=MODEL_CONFIG["max_tokens"]
        )
        
        self.output_parser = StrOutputParser()
        self._setup_chains()
    
    def _setup_chains(self):
        """Set up all LLM chains"""
        SYSTEM_TEMPLATE = "You are a helpful assistant."
        
        # Chain for generating new QA pairs from text differences
        self.new_qa_template = ChatPromptTemplate.from_messages([
            ("system", SYSTEM_TEMPLATE),
            ("user", QA_GENERATE_TEMPLATES)
        ])
        self.new_qa_chain = self.new_qa_template | self.model | self.output_parser
        
        # Chain for updating existing QAs with new content
        self.update_qa_template = ChatPromptTemplate.from_messages([
            ("system", SYSTEM_TEMPLATE),
            ("user", QA_TEMPLATE_WITH_CONTEXT)
        ])
        self.update_qa_chain = self.update_qa_template | self.model | self.output_parser
        
        # QA quality check chain
        self.qa_check_template = ChatPromptTemplate.from_messages([
            ("system", QA_CHECK_SYSTEM_TEMPLATES),
            ("user", QA_CHECK_USER_TEMPLATES)
        ])
        self.qa_check_chain = (
            self.qa_check_template 
            | self.model 
            | self.output_parser 
            | self._parse_check_output
        )
        
        # Answer similarity check chain
        self.similarity_check_template = ChatPromptTemplate.from_messages([
            ("system", SYSTEM_TEMPLATE),
            ("user", SAME_ANSWER_CHECK_TEMPLATE)
        ])
        self.similarity_check_chain = (
            self.similarity_check_template 
            | self.model 
            | self.output_parser 
            | self._parse_similarity_check_output
        )
    
    def _parse_check_output(self, output: str) -> bool:
        return "yes" in output.lower()
    
    def _parse_similarity_check_output(self, output: str) -> bool:
        return "yes" in output.lower()
    
    def _parse_qa_output(self, output: str) -> Tuple[Optional[str], Optional[str], Optional[str]]:
        """Parse the QA generation output"""
        # Remove any newlines and extra spaces
        output = ' '.join(output.split())
        
        # Main pattern matching the expected format
        pattern = r"\{Question:\s*(.+?)\}\s*\{Old Answer:\s*(.+?)\}\s*\{New Answer:\s*(.+?)\}"
        match = re.search(pattern, output)
        
        if match:
            question, answer_old, answer_new = match.groups()
            # Clean up the extracted text
            question = question.strip()
            answer_old = answer_old.strip()
            answer_new = answer_new.strip()
            
            # Basic validation
            if all(len(x) > 0 for x in [question, answer_old, answer_new]):
                return question, answer_old, answer_new
        
        # If no valid format is found, log the output for debugging
        logging.warning(f"Failed to parse output: {output}")
        return None, None, None
    
    def update_qa(self, question: str, old_answer: str, old_sentence: str,
                new_sentence: str, old_content: str, new_content: str,
                max_retries: int = 3) -> ProcessingResult:
        """Update existing QA with new content"""
        example = (
            f"# Question\n{question}\n"
            f"# Sentence(s)\n{old_sentence}\n"
            f"# Source Content\n{old_content}\n"
            f"# Answer\n{old_answer}"
        )
        
        for _ in range(max_retries):
            try:
                # Generate new answer
                output = self.update_qa_chain.invoke({
                    "example": example,
                    "question": question,
                    "sentence": new_sentence,
                    "source_content": new_content
                })
                
                # Check answer quality
                if not self.qa_check_chain.invoke({
                    "question": question,
                    "answer": output,
                    "context": new_content
                }):
                    continue
                
                # Check if answer is semantically different
                if not self.similarity_check_chain.invoke({
                    "question": question,
                    "answer_old": old_answer,
                    "answer_new": output
                }):
                    return ProcessingResult(
                        success=True,
                        question=question,
                        answer_old=old_answer,
                        answer_new=output
                    )
            
            except Exception as e:
                return ProcessingResult(
                    success=False,
                    error=str(e)
                )
        
        return ProcessingResult(success=False)
    
    def update_qa_(self, input: ProcessingInput) -> ProcessingResult:
        return self.update_qa(
            question=input.question,
            old_answer=input.old_answer,
            old_sentence=input.old_sentence,
            new_sentence=input.new_sentence,
            old_content=input.old_content,
            new_content=input.new_content
        )
    
    def generate_qa(self, old_sentence: str, new_sentence: str,
                  old_content: str, new_content: str, source_content: str,
                  max_retries: int = 3) -> ProcessingResult:
        """Generate new QA pair from text differences"""
        for _ in range(max_retries):
            try:
                # Generate QA pair
                output = self.new_qa_chain.invoke({
                    "old_sentence": old_sentence,
                    "new_sentence": new_sentence,
                    "source_content": source_content
                })
                
                # Parse output
                question, answer_old, answer_new = self._parse_qa_output(output)
                if not all([question, answer_old, answer_new]):
                    continue
                
                # Check answer quality
                if not (self.qa_check_chain.invoke({
                    "question": question,
                    "answer": answer_old,
                    "context": old_content
                }) and self.qa_check_chain.invoke({
                    "question": question,
                    "answer": answer_new,
                    "context": new_content
                })):
                    continue
                
                # Check if answers are semantically different
                if not self.similarity_check_chain.invoke({
                    "question": question,
                    "answer_old": answer_old,
                    "answer_new": answer_new
                }):
                    return ProcessingResult(
                        success=True,
                        question=question,
                        answer_old=answer_old,
                        answer_new=answer_new
                    )
            
            except Exception as e:
                return ProcessingResult(
                    success=False,
                    error=str(e)
                )
        
        return ProcessingResult(success=False)

    def generate_qa_(self, input: ProcessingInput) -> ProcessingResult:
        return self.generate_qa(
            old_sentence=input.old_sentence,
            new_sentence=input.new_sentence,
            old_content=input.old_content,
            new_content=input.new_content,
            source_content=input.source_content
        )
    
    def update_qa_batch(self, inputs: List[ProcessingInput]) -> List[ProcessingResult]:
        return RunnableLambda(self.update_qa_).batch(inputs)
    
    def generate_qa_batch(self, inputs: List[ProcessingInput]) -> List[ProcessingResult]:
        return RunnableLambda(self.generate_qa_).batch(inputs)