# RFP AI Agents

The **RFP AI Agents** project is designed to simplify and automate the process of responding to Request for Proposals (RFPs). The AI-powered system analyzes uploaded RFP files, extracts key information, matches them with historical answers, and generates personalized, accurate responses.

## Key Features
- **Multi-format File Support:** Process RFPs in TXT, DOCX, or PDF formats.
- **Intelligent Data Extraction:** Extract details like title, company name, sender's email, requirements, and project scope.
- **Historical Answer Lookup:** Match new RFP questions with answers stored in various Excel files.
- **Automated Responses:** Generate precise and context-aware responses for RFP questions.

---

## Project Installation



1. git clone this repo and 
cd into rfp_ai_agents
2. setup your environment usig poetry or pip: run poetry install / pip install -r requirements.txt

3. Setup your envirnment varibles in .env file as:
OPENAI_API_KEY=<your-openai-api-key>


## Usage
### Prepare Historical Data:

Place Excel files with historical RFP answers in the data/ directory.
Ensure files contain columns relevant to matching RFP questions.
Upload RFP Files:

Place new RFP files (TXT, DOCX, PDF) in the rfp_data/ directory.
## Run the Application:
run the command *python main.py*