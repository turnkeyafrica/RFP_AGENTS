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
            max_tokens=1500
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
            results = vectorstore.similarity_search_with_score(question, k=2)
            
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
            goal="Validate answers and use them to write a Response to the question",
            backstory="Expert at evaluating answer quality and relevance to original questions",
            max_iter=2,
            max_tokens=4000
        )

    async def execute(self, task: Dict[str, Any]) -> Dict[str, Any]:
        original_question = task.get('original_question')
        answers = task.get('answers')
        
        filtered_answers = self._filter_best_answers(answers)
        
        prompt = f"""
        Evaluate these answers for the RFP question:
        Original Question: {original_question}

        Available answers:
        {self._format_answers(filtered_answers)}
        
        Tasks:
        1. Create a comprehensive response combining the most relevant information (about 6 sentences)
        2. List any missing aspects that need more information

        Format:
        COMPREHENSIVE_ANSWER: [Your response]
        RECOMMENDATIONS: [List missing aspects or write "None required" if complete]
        """
        
        response = await self.get_completion(prompt)
        evaluation = self._parse_evaluation(response)
        
        return {
            'original_question': original_question,
            'final_response': evaluation['comprehensive_answer'],
            'recommendations': evaluation['recommendations'],
            'sub_questions': list(answers.keys()),
            'all_answers': filtered_answers
        }

    def _filter_best_answers(self, answers: Dict[str, Any]) -> Dict[str, Any]:
        """Filter to keep only the most relevant answers based on similarity score"""
        filtered = {}
        for question, answer_list in answers.items():
            # Sort by similarity score and keep only the best answer
            sorted_answers = sorted(answer_list, 
                                 key=lambda x: x['similarity_score'], 
                                 reverse=True)
            filtered[question] = sorted_answers[:1] 
        return filtered

    def _parse_evaluation(self, response: str) -> Dict[str, Any]:
        sections = {
            'comprehensive_answer': '',
            'recommendations': []
        }
        
        current_section = None
        for line in response.split('\n'):
            line = line.strip()
            if not line:
                continue
                
            if line.startswith('COMPREHENSIVE_ANSWER:'):
                current_section = 'comprehensive_answer'
                sections['comprehensive_answer'] = line.replace('COMPREHENSIVE_ANSWER:', '').strip()
            elif line.startswith('RECOMMENDATIONS:'):
                current_section = 'recommendations'
                recommendations = line.replace('RECOMMENDATIONS:', '').strip()
                sections['recommendations'] = [r.strip() for r in recommendations.split(',') if r.strip()]
            elif current_section == 'comprehensive_answer':
                sections['comprehensive_answer'] += ' ' + line
                
        return sections

    def _format_answers(self, answers: Dict[str, Any]) -> str:
        """Format answers concisely for the audit prompt"""
        formatted = []
        for question, answer_list in answers.items():
            formatted.append(f"\nQ: {question}")
            for answer_data in answer_list:
                formatted.append(f"A: {answer_data['answer']}")
                formatted.append(f"Score: {answer_data['similarity_score']:.2f}")
        return "\n".join(formatted)   
     
class AgentExecutor:
    async def execute_rfp_processing(self, questions: List[str], 
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
                    'vectorstore': vectorstore
                }
                answers = await lookup_agent.execute(lookup_task)
                
                # Step 3: Audit answers
                audit_task = {
                    'original_question': question,
                    'answers': answers['answers'],
                    'vectorstore': vectorstore
                }
                final_result = await audit_agent.execute(audit_task)
                
                results.append(final_result)
            except Exception as e:
                print(f"Error processing question '{question}': {str(e)}")
                continue
            
        return results

    async def shutdown(self):
        pass