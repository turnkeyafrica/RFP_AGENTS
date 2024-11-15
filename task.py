from dataclasses import dataclass
from typing import Dict, Any, Optional, List
from langchain_community.vectorstores import Chroma
from langchain.docstore.document import Document

@dataclass
class TaskConfig:
    description: str
    expected_output: str
    async_execution: bool = True
    output_file: Optional[str] = None
    search_kwargs: Dict[str, Any] = None

    def __post_init__(self):
        if self.search_kwargs is None:
            self.search_kwargs = {
                "k": 2,
                "score_threshold": 0.7
            }

class BaseTask:
    def __init__(self, config: TaskConfig):
        self.config = config

    def to_dict(self) -> Dict[str, Any]:
        return {
            "description": self.config.description,
            "expected_output": self.config.expected_output,
            "async_execution": self.config.async_execution,
            "output_file": self.config.output_file,
            "search_kwargs": self.config.search_kwargs
        }

class Generate3QuestionsTask(BaseTask):
    def __init__(self, original_question: str):
        config = TaskConfig(
            description=f"""
                Break down the following RFP question into specific sub-questions:
                Original Question: {original_question}
                
                Requirements:
                - Generate three detailed sub-questions
                - Each sub-question should focus on a different aspect
                - Questions should be more specific than the original
                - Questions should help gather comprehensive information
                - Output should be clear and answerable
            """,
            expected_output="Three specific sub-questions derived from the original question",
            async_execution=True
        )
        super().__init__(config)
        self.original_question = original_question
        
    def validate_questions(self, questions: List[str]) -> List[str]:
        """
        Validate and format the generated questions
        """
        # Remove any numbering or bullets
        cleaned_questions = [
            q.lstrip('0123456789.). -')
            for q in questions
            if q.strip()
        ]
        
        # Ensure we have exactly 3 questions
        if len(cleaned_questions) > 3:
            cleaned_questions = cleaned_questions[:3]
        elif len(cleaned_questions) < 3:
            # Pad with more specific versions if we don't have enough
            while len(cleaned_questions) < 3:
                cleaned_questions.append(
                    f"What specific details about {self.original_question.lower()} "
                    f"are most relevant for question {len(cleaned_questions) + 1}?"
                )
                
        return cleaned_questions

class LookupAnswersTask(BaseTask):
    def __init__(
        self, 
        original_question: str, 
        sub_questions: List[str], 
        qa_database_path: str,
        vectorstore: Optional[Chroma] = None,
        search_kwargs: Optional[Dict[str, Any]] = None
    ):
        if search_kwargs is None:
            search_kwargs = {
                "k": 2,
                "score_threshold": 0.7,
            }
            
        config = TaskConfig(
            description=f"""
                Find the best matching answers using semantic search for the generated sub-questions:
                Original Question: {original_question}
                Sub-questions: {', '.join(sub_questions)}
                
                Search Parameters:
                - Similarity threshold: {search_kwargs.get('score_threshold', 0.7)}
                - Results per question: {search_kwargs.get('k', 2)}
            """,
            expected_output="Ranked answers with similarity scores for each sub-question",
            async_execution=True,
            search_kwargs=search_kwargs
        )
        super().__init__(config)
        self.original_question = original_question
        self.sub_questions = sub_questions
        self.qa_database_path = qa_database_path
        self.vectorstore = vectorstore

    async def semantic_search(self, query: str) -> List[Dict[str, Any]]:
        """
        Perform semantic search using Chroma
        """
        if not self.vectorstore:
            raise ValueError("Vector store not initialized")
            
        results = self.vectorstore.similarity_search_with_relevance_scores(
            query,
            k=self.config.search_kwargs.get('k', 2)
        )
        
        return [
            {
                "content": doc.page_content,
                "score": score,
                "metadata": doc.metadata
            }
            for doc, score in results
        ]

class AuditAnswersTask(BaseTask):
    def __init__(
        self, 
        original_question: str, 
        answers: Dict[str, Dict[str, Any]], 
        qa_database_path: str,
        vectorstore: Optional[Chroma] = None
    ):
        config = TaskConfig(
            description=f"""
                Audit the semantic search results for the RFP question:
                Original Question: {original_question}
                
                Requirements:
                - Evaluate answer relevance and comprehensiveness
                - Combine relevant information from multiple sources
                - Assess answer quality using similarity scores
                - Identify any gaps in information coverage
                - Provide specific recommendations for improvement
                - Ensure answer clarity and completeness
            """,
            expected_output="Comprehensive answer with improvement recommendations",
            async_execution=True
        )
        super().__init__(config)
        self.original_question = original_question
        self.answers = answers
        self.qa_database_path = qa_database_path
        self.vectorstore = vectorstore
        
    def evaluate_answer_quality(self, answer: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluate the quality of an answer based on similarity scores and content
        """
        return {
            "content": answer["content"],
            "quality_score": answer["score"],
            "has_sufficient_detail": answer["score"] > self.config.search_kwargs.get("score_threshold", 0.7),
            "metadata": answer.get("metadata", {})
        }
        
    def generate_improvement_recommendations(self, answers: Dict[str, List[Dict[str, Any]]]) -> List[str]:
        """
        Generate recommendations for improving answer quality
        """
        recommendations = []
        for question, answer_list in answers.items():
            low_quality_answers = [
                answer for answer in answer_list 
                if answer["score"] < self.config.search_kwargs.get("score_threshold", 0.7)
            ]
            
            if low_quality_answers:
                recommendations.append(
                    f"Consider expanding knowledge base for question: {question}"
                )
                
        return recommendations