import asyncio
import os
from dotenv import load_dotenv
import time
from agents import Generate3QuestionsAgent, LookupAgent, AuditAgent, AgentExecutor
import pdfplumber
from docx import Document
import pandas as pd
from typing import List, Dict, Any
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

rfp_data_directory = 'rfp_data'
qa_database_file = 'data/qa_database.xlsx'
chroma_persist_directory = "chroma_db"

def load_rfp_questions_from_directory(directory: str) -> List[str]:
    questions = []
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        
        if filename.endswith(".pdf"):
            with pdfplumber.open(file_path) as pdf:
                for page in pdf.pages:
                    text = page.extract_text()
                    questions.extend(text.splitlines())
        
        elif filename.endswith(".docx"):
            doc = Document(file_path)
            for paragraph in doc.paragraphs:
                questions.append(paragraph.text)
        
        elif filename.endswith(".txt"):
            with open(file_path, 'r', encoding='utf-8') as txt_file:
                text = txt_file.read()
                questions.extend(text.splitlines())
    
    return questions

rfp_questions = load_rfp_questions_from_directory(rfp_data_directory)

def setup_chroma_database(qa_df: pd.DataFrame) -> Chroma:
    """
    Set up Chroma database from QA pairs in the Excel file
    """
    qa_texts = [
        f"Question: {row['Question']}\n Answer: {row['Answer']}"
        for _, row in qa_df.iterrows()
    ]
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=0,
        length_function=len,
    )
    
    chunks = text_splitter.create_documents(qa_texts)
    embeddings = OpenAIEmbeddings()
    
    # Create Chroma database with persistence
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=chroma_persist_directory
    )
    vectorstore.persist()
    
    return vectorstore

def load_chroma_database() -> Chroma:
    """
    Load the existing Chroma database for semantic search.
    """
    embeddings = OpenAIEmbeddings()
    
    if os.path.exists(chroma_persist_directory):
        print("Loading existing Chroma database...")
        return Chroma(
            persist_directory=chroma_persist_directory,
            embedding_function=embeddings
        )
    else:
        raise ValueError(f"Error: Chroma database not found at '{chroma_persist_directory}'. Please create it first.")

async def main():
    if not os.path.exists(rfp_data_directory):
        os.makedirs(rfp_data_directory)
    
    if not os.path.exists(os.path.dirname(qa_database_file)):
        os.makedirs(os.path.dirname(qa_database_file))
    
    try:
        qa_df = pd.read_excel(qa_database_file)
        
        try:
            vectorstore = load_chroma_database()
            print("Loaded existing Chroma database")
        except ValueError:
            vectorstore = setup_chroma_database(qa_df)
            print("Created new Chroma database")
            
        print("Chroma database ready for semantic search")
    except FileNotFoundError:
        print(f"QA database file not found at '{qa_database_file}'. Ensure the correct path.")
        return
    except Exception as e:
        print(f"Error setting up Chroma database: {str(e)}")
        return
    
    
    executor = AgentExecutor()
    
    start_time = time.time()
    
    try:
        results = await executor.execute_rfp_processing(
            questions=rfp_questions,
            vectorstore=vectorstore
        )
        
        print("\nRFP Processing Results:")
        for i, result in enumerate(results, 1):
            print(f"\nQuestion {i}: {result['original_question']}")
            print("\nGenerated Sub-questions:")
            for j, question in enumerate(result['sub_questions'], 1):
                print(f"{j}. {question}")
            print("\n Final Answer:")
            print(result['final_response'])
            if result['recommendations']:
                print("\nRecommendations for Improvement:")
                for rec in result['recommendations']:
                    print(f"- {rec}")

    except Exception as e:
        print(f"Error during execution: {str(e)}")
        print(f"Full error details: ", e.__class__.__name__)
    finally:
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"\nExecution took {elapsed_time:.2f} seconds")
        await executor.shutdown()

if __name__ == "__main__":
    asyncio.run(main())