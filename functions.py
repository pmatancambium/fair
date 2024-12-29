import os
from datetime import datetime
import json
from pymongo import MongoClient
from google import genai
import google.generativeai as genaiEmb
from google.genai.types import Tool, GenerateContentConfig, GoogleSearch

# from dotenv import load_dotenv
import streamlit as st

# Load environment variables
# load_dotenv()

# Configure API keys and clients
# GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
# MONGODB_URI = os.getenv("MONGODB_URI")
GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
MONGODB_URI = st.secrets["MONGODB_URI"]

genaiEmb.configure(api_key=GEMINI_API_KEY)
client = genai.Client(api_key=GEMINI_API_KEY)
model_id = "gemini-2.0-flash-exp"

# Initialize Google Search tool
google_search_tool = Tool(google_search=GoogleSearch())


def init_mongodb():
    """Initialize MongoDB connection"""
    client = MongoClient(MONGODB_URI)
    return client.conversations_db.conversations


def get_embedding(text):
    """Get text embedding using Gemini API"""
    result = genaiEmb.embed_content(
        model="models/text-embedding-004", content=text, task_type="retrieval_query"
    )
    return result["embedding"]


def get_all_clients(collection):
    """Get sorted list of all client IDs"""
    clients = collection.distinct("contact_id")
    return sorted([int(cid) for cid in clients])


def find_similar_conversations(collection, query_embedding, contact_id, n=10):
    """Find similar conversations using vector search"""
    contact_id = int(contact_id)
    results = collection.aggregate(
        [
            {
                "$vectorSearch": {
                    "index": "vector_index",
                    "path": "embedding",
                    "queryVector": query_embedding,
                    "numCandidates": 100,
                    "limit": n,
                    "filter": {"contact_id": {"$eq": contact_id}},
                }
            },
            {
                "$project": {
                    "conversation_id": 1,
                    "contact_id": 1,
                    "start_time": 1,
                    "end_time": 1,
                    "messages": 1,
                    "text_for_embedding": 1,
                    "search_score": {"$meta": "vectorSearchScore"},
                }
            },
        ]
    )
    return list(results)


def format_context(conversations):
    """Format conversations for context"""
    formatted_context = []
    for conv in conversations:
        context = {
            "conversation_id": conv["conversation_id"],
            "timestamp": datetime.fromtimestamp(conv["start_time"] / 1000).strftime(
                "%Y-%m-%d %H:%M:%S"
            ),
            "similarity_score": round(conv["search_score"], 3),
            "conversation": [],
        }

        for msg in conv["messages"]:
            if msg.get("quick_replies"):
                continue
            context["conversation"].append(
                {
                    "role": msg["role"],
                    "content": msg["content"],
                }
            )

        formatted_context.append(context)

    return formatted_context


def get_gemini_response(question, context, conversation_history=[]):
    """Get response from Gemini model with conversation history"""
    # Format conversation history
    history_text = ""
    if conversation_history:
        history_text = "Previous conversation:\n"
        for msg in conversation_history:
            role = "User" if msg["role"] == "user" else "Assistant"
            history_text += f"{role}: {msg['content']}\n"
        history_text += "\n"

    prompt = f"""
בהתבסס על ארכיון השיחות עם הלקוח שלנו ב-'fair: קרנות נאמנות אונליין'
    {json.dumps(context, indent=2, ensure_ascii=False)}

{history_text}
Current question: {question}

אם הקונטקסט לא מכיל מידע רלוונטי - תאמר זאת.
    """

    try:
        response = client.models.generate_content(
            model=model_id,
            contents=prompt,
            config=GenerateContentConfig(
                # tools=[google_search_tool],
                response_modalities=["TEXT"],
            ),
        )

        # Safely extract search entry point if it exists
        search_entry_point = None
        try:
            if (
                response.candidates
                and hasattr(response.candidates[0], "grounding_metadata")
                and hasattr(
                    response.candidates[0].grounding_metadata, "search_entry_point"
                )
                and hasattr(
                    response.candidates[0].grounding_metadata.search_entry_point,
                    "rendered_content",
                )
            ):
                search_entry_point = response.candidates[
                    0
                ].grounding_metadata.search_entry_point.rendered_content
        except AttributeError:
            search_entry_point = None

        # Extract response text
        if hasattr(response, "text"):
            response_text = response.text
        elif isinstance(response, tuple):
            response_text = str(response[0])
        else:
            response_text = str(response)

        return {"text": response_text, "search_entry_point": search_entry_point}

    except Exception as e:
        print(f"Error generating response: {e}")
        return {"text": f"Error generating response: {e}", "search_entry_point": None}
