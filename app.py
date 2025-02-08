from flask import Flask, request, jsonify
from huggingface_hub import InferenceClient
from db import db  # Import the database connection from your db.py
from flask_cors import CORS
import json
import re
from dotenv import load_dotenv
import os
load_dotenv()

# Initialize the Hugging Face InferenceClient with your API key.
client = InferenceClient(
    provider="hf-inference",
    api_key=os.getenv("HF_API_KEY")
)

# Common database schema (this text is inserted in the prompt)
SCHEMA_TEXT = """
Teams Collection Schema:
  - team_name (string): Name of the team.
  - members (array of strings): List of team members.
  - project (string): Title of the project.
  - category (string): Category of the project.
  - submission_time (Date): When the project was submitted.

Judges Collection Schema:
  - name (string): Judge name (required).
  - expertise (string): Judge expertise (required).

Scores Collection Schema:
  - score_id (INTEGER): Unique score identifier.
  - team_id (INTEGER): References a team.
  - judge_id (INTEGER): References a judge.
  - score (FLOAT): The score value.
  - comments (string, optional): Additional comments.
"""

def get_db_context():
    """
    Fetch data from the 'teams', 'judges', and 'scores' collections.
    """
    context_parts = []
    try:
        # For debugging: list available collections.
        available_collections = db.list_collection_names()
        context_parts.append(f"Available Collections: {available_collections}")

        teams = list(db.teams.find({}, {"_id": 0}))
        judges = list(db.judges.find({}, {"_id": 0}))
        scores = list(db.scores.find({}, {"_id": 0}))

        if teams:
            teams_str = "\n".join(str(team) for team in teams)
            context_parts.append("Teams Data:\n" + teams_str)
        if judges:
            judges_str = "\n".join(str(judge) for judge in judges)
            context_parts.append("Judges Data:\n" + judges_str)
        if scores:
            scores_str = "\n".join(str(score) for score in scores)
            context_parts.append("Scores Data:\n" + scores_str)
    except Exception as e:
        context_parts.append("Error accessing database: " + str(e))
    return "\n\n".join(context_parts)

# Define the planning prompt template.
PLANNING_PROMPT = """
User Query: "{query}"
Chat History: {chat_history}

{schema}

Database Records:
{db_context}

Instructions:
- Think step by step about whether additional information is needed.
- If you need to query the database, output a JSON object in compact format (i.e. on one line, with no extra spaces or line breaks) with keys:
    "action": "query" and "db_query": "[the query to run]".
- If no additional query is needed, output a JSON object with "action": "none".
- Use ONLY the team names exactly as provided in the database records. Do not invent names such as "alpha", "beta", etc.
- Output ONLY the JSON object (do not include any other text or chain-of-thought).
Plan:
"""

# Define the final answer prompt template.
FINAL_PROMPT = """
User Query: "{query}"
Chat History: {chat_history}

Plan and Database Result:
Plan: {plan}
Database Result: {db_result}

Instructions:
Based on the plan and database result, provide ONLY the final answer in this format:
  
  **Answer:** [Concise summary here]

Do NOT include any internal reasoning or <think> blocks.
Answer:
"""

# Helper function to parse the planning response.
def parse_plan_response(response_text):
    # Remove any <think>...</think> blocks
    cleaned = re.sub(r"<think>[\s\S]*?</think>", "", response_text).strip()
    # Remove newline characters to compact the JSON string
    cleaned = cleaned.replace("\n", " ").strip()
    # Extract JSON object using regex
    match = re.search(r"({.*})", cleaned)
    if match:
        json_text = match.group(1)
        try:
            plan_obj = json.loads(json_text)
            return plan_obj
        except Exception as e:
            print("JSON parsing error:", e)
    return {"action": "none"}

# Create a Flask application.
app = Flask(__name__)
CORS(app, resources={r'/*': {'origins': '*'}})

# Debug route to verify database connection and collections.
@app.route('/debug', methods=['GET'])
def debug():
    try:
        collections = db.list_collection_names()
        samples = {}
        for coll in collections:
            sample_doc = db[coll].find_one({}, {"_id": 0})
            samples[coll] = sample_doc
        return jsonify({"collections": collections, "samples": samples})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/query', methods=['POST'])
def query():
    try:
        data = request.get_json(force=True)
        user_query = data.get('query', '')
        chat_history = data.get('chat_history', '')
        
        db_context = get_db_context()

        # Step 1: Generate a plan.
        planning_prompt = PLANNING_PROMPT.format(
            query=user_query,
            chat_history=chat_history,
            schema=SCHEMA_TEXT,
            db_context=db_context
        )
        planning_messages = [{"role": "user", "content": planning_prompt}]
        planning_completion = client.chat.completions.create(
            model="deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
            messages=planning_messages,
            max_tokens=2048,
            temperature=0.3
        )
        planning_response = planning_completion.choices[0].message.content.strip()
        # print("Planning Response:\n", planning_response)

        # Parse the JSON plan from the planning response.
        plan_obj = parse_plan_response(planning_response)
        # print("Parsed Plan:", plan_obj)

        # Step 2: If plan indicates a database query, execute it.
        if plan_obj.get("action", "none") == "query":
            db_query = plan_obj.get("db_query", "")
            # For this example, we assume db_query specifies the team name to look up.
            # (In practice, you'd need to design a way to parse and execute various queries.)
            db_result = "No result"
            if db_query:
                result = db.teams.find_one({"team_name": db_query}, {"_id": 0})
                db_result = str(result) if result else "No matching record found."
        else:
            db_result = "No database query executed."

        # Step 3: Generate the final answer.
        final_prompt = FINAL_PROMPT.format(
            query=user_query,
            chat_history=chat_history,
            plan=planning_response,
            db_result=db_result
        )
        final_messages = [{"role": "user", "content": final_prompt}]
        final_completion = client.chat.completions.create(
            model="deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
            messages=final_messages,
            max_tokens=500,
            temperature=0.3
        )
        final_response = final_completion.choices[0].message.content.strip()
        # print("Final Response:\n", final_response)
        return jsonify({"response": final_response})
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    # app.run(debug=True) #for production
    app.run()