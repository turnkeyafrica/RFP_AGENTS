import os
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import asyncio
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
import pandas as pd
from abc import ABC, abstractmethod
from langchain_community.vectorstores import Chroma
import tiktoken


@dataclass
class AgentConfig:
    role: str
    goal: str
    backstory: str
    verbose: bool = True
    max_iter: int = 2
    max_tokens: int = 7800


def count_tokens(text: str) -> int:
    """Count tokens in a text string using GPT-4 encoding"""
    encoding = tiktoken.encoding_for_model("gpt-4")
    return len(encoding.encode(text))

def truncate_text(text: str, max_tokens: int) -> str:
    """Truncate text to fit within max_tokens"""
    encoding = tiktoken.encoding_for_model("gpt-4")
    tokens = encoding.encode(text)
    if len(tokens) <= max_tokens:
        return text
    return encoding.decode(tokens[:max_tokens])

class BaseAgent(ABC):
    def __init__(self, llm: Optional[Any] = None):
        self.llm = llm if llm else ChatOpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
            model="gpt-4",
            max_tokens=7800
        )
        self.config: Optional[AgentConfig] = None
    
    @abstractmethod
    async def execute(self, task: Dict[str, Any]) -> str:
        pass

    async def get_completion(self, prompt: str) -> str:
        truncated_prompt = truncate_text(prompt, self.config.max_tokens - 1000)
        messages = [HumanMessage(content=truncated_prompt)]
        response = await self.llm.ainvoke(messages)
        return response.content

class Generate3QuestionsAgent(BaseAgent):
    def __init__(self, llm=None):
        super().__init__(llm)
        self.config = AgentConfig(
            role="Question Generator",
            goal="Generate specific sub-questions from RFP questions",
            backstory="Expert at breaking down complex RFP questions into specific, answerable components",
            max_iter=2,
            max_tokens=2000
        )

    async def execute(self, task: Dict[str, Any]) -> Dict[str, List[str]]:
        original_question = task.get('question')
        
        prompt = f"""
        Break down the following RFP question into three specific sub-questions that will help gather comprehensive information:
        
        Original Question: {original_question}
        
        Generate three detailed questions that:
        1. Are more specific than the original question
        2. Focus on different aspects of the information needed
        3. Will help provide a complete answer to the original question
        
        Format the response as three numbered questions only.
        """
        
        response = await self.get_completion(prompt)
        questions = [q.strip() for q in response.split('\n') if q.strip()]
        questions = [q.lstrip('0123456789.). ') for q in questions]
        print(questions)
        return {
            'original_question': original_question,
            'sub_questions': questions[:3]
        }

class LookupAgent(BaseAgent):
    def __init__(self, llm=None):
        super().__init__(llm)
        self.config = AgentConfig(
            role="Answer Lookup Specialist",
            goal="Find best matching answers from historical Q&A chroma database",
            backstory="Expert at semantic search and answer matching from existing knowledge base",
            max_iter=2,
            max_tokens=3000
        )
        

    async def execute(self, task: Dict[str, Any]) -> Dict[str, Any]:
        questions = task.get('sub_questions')
        vectorstore = task.get('vectorstore')
        
        all_answers = {}
        for question in questions:
            # Use Chroma's similarity search
            results = vectorstore.similarity_search_with_score(question, k=2)
            
            # Format the results
            formatted_answers = []
            for doc, score in results:
                truncated_answer = truncate_text(doc.page_content, 300)
                formatted_answers.append({
                    'answer': truncated_answer,
                    'similarity_score': score
                })
            
            all_answers[question] = formatted_answers
            print(all_answers)
        return {
            'original_question': task.get('original_question'),
            'sub_questions': questions,
            'answers': all_answers
        }

class AuditAgent(BaseAgent):
    def __init__(self, llm=None):
        super().__init__(llm)
        self.config = AgentConfig(
            role="Answer Quality Auditor",
            goal="Validate answers and provide recommendations for RFP questions",
            backstory="Expert at evaluating answer quality and relevance to original questions",
            max_iter=2,
            max_tokens=4800
        )

    async def execute(self, task: Dict[str, Any]) -> Dict[str, Any]:
        original_question = task.get('original_question')
        answers = task.get('answers')
        
        prompt = f"""
        You are evaluating answers for this RFP question:
        Original Question: {original_question}

        Review these sub-questions and their answers:
        {self._format_answers(answers)}
        
        Your task is to:
        1. Evaluate the relevance and completeness of the answers
        2. Combine the most relevant points to create a comprehensive response
        3. Provide specific recommendations if any important aspects are not adequately addressed


        Format your response exactly as follows:
        COMPREHENSIVE_ANSWER: [Provide a well-structured answer combining the best elements in 2-3 sentences]
        RECOMMENDATIONS: [If any aspects need improvement, list specific recommendations. If none needed, write "None required"]
        """
        
        response = await self.get_completion(prompt)
        evaluation = self._parse_evaluation(response)
        
        return {
            'original_question': original_question,
            'final_answer': evaluation['comprehensive_answer'],
            'recommendations': evaluation['recommendations'],
            'sub_questions': list(answers.keys()),
            'all_answers': answers
        }

    def _parse_evaluation(self, response: str) -> Dict[str, Any]:
        """Parse the evaluation response"""
        lines = response.split('\n')
        evaluation = {
            'comprehensive_answer': '',
            'recommendations': []
        }
        
        current_section = None
        for line in lines:
            if line.startswith('COMPREHENSIVE_ANSWER:'):
                current_section = 'comprehensive_answer'
                evaluation['comprehensive_answer'] = line.replace('COMPREHENSIVE_ANSWER:', '').strip()
            elif line.startswith('RECOMMENDATIONS:'):
                current_section = 'recommendations'
                recommendations = line.replace('RECOMMENDATIONS:', '').strip()
                evaluation['recommendations'] = [r.strip() for r in recommendations.split(',') if r.strip()]
                
        return evaluation

    def _format_answers(self, answers: Dict[str, Any]) -> str:
        """Format answers for the audit prompt"""
        formatted = []
        for question, answer_list in answers.items():
            formatted.append(f"\nSub-question: {question}")
            formatted.append("Answers:")
            for i, answer_data in enumerate(answer_list, 1):
                formatted.append(f"{i}. {answer_data['answer']}")
                formatted.append(f"   Similarity Score: {answer_data['similarity_score']:.3f}")
                
        return "\n".join(formatted)
    
class AgentExecutor:
    async def execute_rfp_processing(self, questions: List[str], qa_database_path: str, 
                                   vectorstore: Chroma) -> List[Dict[str, Any]]:
        generate_agent = Generate3QuestionsAgent()
        lookup_agent = LookupAgent()
        audit_agent = AuditAgent()
        
        results = []
        
        for question in questions:
            try:
                # Step 1: Generate sub-questions
                gen_task = {'question': question}
                sub_questions = await generate_agent.execute(gen_task)
                
                # Step 2: Look up answers
                lookup_task = {
                    'original_question': question,
                    'sub_questions': sub_questions['sub_questions'],
                    'qa_database_path': qa_database_path,
                    'vectorstore': vectorstore
                }
                answers = await lookup_agent.execute(lookup_task)
                
                # Step 3: Audit answers
                audit_task = {
                    'original_question': question,
                    'answers': answers['answers'],
                    'qa_database_path': qa_database_path,
                    'vectorstore': vectorstore
                }
                final_result = await audit_agent.execute(audit_task)
                
                results.append(final_result)
            except Exception as e:
                print(f"Error processing question '{question}': {str(e)}")
                continue
            
        return results

    async def shutdown(self):
        # Cleanup code if needed
        pass