import streamlit as st
from dotenv import load_dotenv
import os
from datetime import datetime
import html
from functions import (
    init_mongodb,
    get_all_clients,
    get_embedding,
    find_similar_conversations,
    format_context,
    get_gemini_response,
)

# Load environment variables
load_dotenv()

# Configure Streamlit page settings
st.set_page_config(
    page_title="ניתוח צ'אט תמיכת לקוחות",
    page_icon="💬",
    layout="wide",
)

# Add custom CSS for RTL support
st.markdown(
    """
<style>
    /* RTL support for main content */
    .main, .element-container, .stMarkdown, .stText {
        direction: rtl;
        text-align: right;
    }

    /* RTL support for selectbox and input */
    .stSelectbox, .stTextInput {
        direction: rtl;
        text-align: right;
    }

    /* RTL support for chat messages */
    .stChatMessage {
        direction: rtl;
        text-align: right;
    }

    /* Keep English text LTR within RTL context */
    .ltr {
        direction: ltr;
        display: inline-block;
    }

    /* Custom styling for chat messages */
    .chat-message {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        max-width: 80%;
    }

    .user-message {
        background-color: #e6f3ff;
        margin-left: auto;
    }

    .assistant-message {
        background-color: #f0f0f0;
        margin-right: auto;
    }

    /* Context display styling */
    .context-container {
        margin: 1rem 0;
        font-size: 0.9rem;
    }

    .context-conversation {
        background-color: #ffffff;
        border: 1px solid #e0e0e0;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }

    .context-header {
        background-color: #f8f9fa;
        padding: 0.5rem 1rem;
        border-bottom: 1px solid #e0e0e0;
        display: flex;
        justify-content: space-between;
        align-items: center;
        border-radius: 0.5rem 0.5rem 0 0;
    }

    .context-timestamp {
        color: #666;
        font-size: 0.85rem;
    }

    .context-score {
        background-color: #e6f3ff;
        padding: 0.2rem 0.5rem;
        border-radius: 1rem;
        font-size: 0.85rem;
        color: #0066cc;
    }

    .context-messages {
        padding: 1rem;
    }

    .context-message {
        margin-bottom: 1rem;
        padding: 0.5rem;
        border-radius: 0.3rem;
    }

    .context-role {
        font-weight: 500;
        margin-bottom: 0.3rem;
        color: #555;
    }

    .user-context {
        background-color: #f8f9fa;
    }

    .assistant-context {
        background-color: #f0f7ff;
    }

    .context-content {
        white-space: pre-wrap;
        word-break: break-word;
    }
/* Common styles for both processed and raw conversations */
    .conversation-container {
        background-color: #ffffff;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin-bottom: 1.5rem;
        overflow: hidden;
    }

    /* Processed Conversations (שיחות מעובדות) Styling */
    .context-container {
        padding: 1rem;
    }

    .context-conversation {
        background-color: #ffffff;
        border: 1px solid #e6e6e6;
        border-radius: 8px;
        margin-bottom: 1.5rem;
        transition: all 0.2s ease;
    }

    .context-conversation:hover {
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }

    .context-header {
        background-color: #f8f9fa;
        padding: 0.75rem 1rem;
        border-bottom: 1px solid #e6e6e6;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }

    .context-timestamp {
        color: #666;
        font-size: 0.9rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }

    .context-score {
        background-color: #e3f2fd;
        padding: 0.4rem 0.8rem;
        border-radius: 20px;
        font-size: 0.9rem;
        color: #1976d2;
        font-weight: 500;
    }

    .context-messages {
        padding: 1rem;
    }

    .context-message {
        margin-bottom: 1rem;
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #eee;
    }

    .context-role {
        font-weight: 600;
        margin-bottom: 0.5rem;
        color: #424242;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }

    .user-context {
        background-color: #f5f5f5;
    }

    .assistant-context {
        background-color: #f3f8ff;
    }

    /* Raw Data (נתונים גולמיים) Styling */
    .raw-context-container {
        padding: 1rem;
    }

    .raw-context-conversation {
        background-color: #ffffff;
        border: 1px solid #e6e6e6;
        border-radius: 8px;
        margin-bottom: 1.5rem;
        transition: all 0.2s ease;
    }

    .raw-context-conversation:hover {
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }

    .raw-context-header {
        background-color: #f8f9fa;
        padding: 1rem;
        border-bottom: 1px solid #e6e6e6;
    }

    .raw-context-meta {
        display: flex;
        justify-content: space-between;
        margin-bottom: 0.5rem;
        font-weight: 500;
    }

    .raw-context-time {
        display: flex;
        justify-content: space-between;
        color: #666;
        font-size: 0.9rem;
    }

    .raw-context-content {
        padding: 1rem;
    }

    .raw-context-text {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 6px;
        margin-bottom: 1rem;
        white-space: pre-wrap;
        font-family: inherit;
        line-height: 1.6;
    }

    .raw-context-score {
        display: flex;
        justify-content: flex-end;
        color: #1976d2;
        font-weight: 500;
    }

    /* Responsive adjustments */
    @media (max-width: 768px) {
        .context-header, .raw-context-meta, .raw-context-time {
            flex-direction: column;
            gap: 0.5rem;
        }

        .context-score {
            margin-top: 0.5rem;
        }
    }

    /* Scrollbar styling */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }

    ::-webkit-scrollbar-track {
        background: #f1f1f1;
        border-radius: 4px;
    }

    ::-webkit-scrollbar-thumb {
        background: #888;
        border-radius: 4px;
    }

    ::-webkit-scrollbar-thumb:hover {
        background: #666;
    }
</style>
""",
    unsafe_allow_html=True,
)

# Initialize session state for persistent storage
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
if "conversation_history" not in st.session_state:
    st.session_state.conversation_history = []
if "selected_client" not in st.session_state:
    st.session_state.selected_client = None


# Password configuration
PASSWORD = st.secrets["APP_PASSWORD"]


def authenticate():
    """Handle password authentication"""
    if st.session_state.authenticated:
        return True

    st.title("🔒 כניסה למערכת")
    password = st.text_input("הכנס סיסמה:", type="password", key="password_input")

    if st.button("כניסה"):
        if password == PASSWORD:
            st.session_state.authenticated = True
            st.rerun()
        else:
            st.error("סיסמה שגויה!")
            return False
    return False


def format_message(text, is_hebrew=True):
    """Format message with appropriate direction"""
    if is_hebrew:
        return f'<div dir="rtl">{text}</div>'
    return f'<div dir="ltr">{text}</div>'


def format_context_display(context):
    """Format conversation context with improved styling"""
    html_output = '<div class="context-container">'

    for conv in context:
        timestamp = conv["timestamp"]
        similarity = conv["similarity_score"]

        html_output += f"""
        <div class="context-conversation">
            <div class="context-header">
                <div class="context-timestamp">
                    <span>🕒</span>
                    <span>{timestamp}</span>
                </div>
                <div class="context-score">
                    <span>התאמה: {similarity}</span>
                </div>
            </div>
            <div class="context-messages">
            <br>
        """

        for msg in conv["conversation"]:
            role = msg["role"].lower()
            is_user = any(
                user_role in role
                for user_role in ["user", "customer", "client", "human"]
            )

            role_class = "user-context" if is_user else "assistant-context"
            role_icon = "👤" if is_user else "🤖"
            role_name = "לקוח" if is_user else "נציג"

            safe_content = html.escape(msg["content"])
            safe_content = safe_content.replace("\\n", "<br>").replace("\n", "<br>")

            html_output += f"""
            <div class="context-message {role_class}">
                <div class="context-role">
                    <span>{role_icon}</span>
                    <span>{role_name}</span>
                </div>
                <div class="context-content">{safe_content}</div>
                <br>
            </div>
            """

        html_output += """
            </div>
        </div>
        """

    html_output += "</div>"
    return html_output


def format_raw_context(similar_convs):
    """Format raw conversation data with improved styling"""
    html_output = '<div class="raw-context-container">'

    for conv in similar_convs:
        text_content = html.escape(conv.get("text_for_embedding", "אין טקסט זמין"))
        text_content = text_content.replace("\\n", "<br>").replace("\n", "<br>")

        start_time = datetime.fromtimestamp(conv["start_time"] / 1000).strftime(
            "%Y-%m-%d %H:%M:%S"
        )
        end_time = datetime.fromtimestamp(conv["end_time"] / 1000).strftime(
            "%Y-%m-%d %H:%M:%S"
        )

        html_output += f"""
        <div class="raw-context-conversation">
            <div class="raw-context-header">
                <div class="raw-context-meta">
                    <span>🆔 שיחה: {conv["conversation_id"]}</span>
                    <span>👤 לקוח: {conv["contact_id"]}</span>
                </div>
                <div class="raw-context-time">
                    <span>⏰ התחלה: {start_time}</span>
                    <span>⌛ סיום: {end_time}</span>
                </div>
            </div>
            <div class="raw-context-content">
                <div class="raw-context-text">
                    <strong>טקסט מלא:</strong><br>
                    {text_content}
                </div>
                <div class="raw-context-score">
                    <span>📊 ציון התאמה: {round(conv["search_score"], 3)}</span>
                </div>
            </div>
        </div>
        """

    html_output += "</div>"
    return html_output


def main():
    # Check authentication before showing the main app
    if not authenticate():
        return
    st.title("💬 ניתוח צ'אט תמיכת לקוחות")

    # Initialize MongoDB connection
    collection = init_mongodb()

    # Sidebar for client selection
    with st.sidebar:
        st.title("בחירת לקוח")
        clients = get_all_clients(collection)
        selected_client = st.selectbox("בחר מזהה לקוח:", clients)

    # Display conversation history
    for message in st.session_state.conversation_history:
        with st.chat_message(message["role"]):
            # Detect if the content contains Hebrew
            has_hebrew = any("\u0590" <= c <= "\u05FF" for c in message["content"])
            st.markdown(
                format_message(message["content"], is_hebrew=has_hebrew),
                unsafe_allow_html=True,
            )

            # Add expander for context
            if "context" in message:
                with st.expander("📜 הצג שיחות קודמות רלוונטיות"):
                    # Display formatted context
                    st.markdown(
                        format_context_display(message["context"]),
                        unsafe_allow_html=True,
                    )

    # Chat input with RTL support
    question = st.chat_input("שאל שאלה על השיחות...")

    if question:
        # Display user message
        with st.chat_message("user"):
            st.markdown(format_message(question), unsafe_allow_html=True)

        # Process the question
        with st.chat_message("assistant"):
            with st.spinner("מעבד..."):
                # Get embeddings and find similar conversations
                query_embedding = get_embedding(question)
                similar_convs = find_similar_conversations(
                    collection, query_embedding, selected_client
                )

                if not similar_convs:
                    response_text = f"לא נמצאו שיחות עבור לקוח {selected_client}"
                    st.warning(format_message(response_text), unsafe_allow_html=True)
                else:
                    # Format context and get response
                    context = format_context(similar_convs)
                    response = get_gemini_response(
                        question, context, st.session_state.conversation_history
                    )

                    # Create tabs for different views
                    response_tab, context_tab, raw_tab = st.tabs(
                        ["✍️ תשובה", "💬 שיחות מעובדות", "📊 נתונים גולמיים"]
                    )

                    with response_tab:
                        # Display the main response text with RTL support
                        st.markdown(
                            format_message(response["text"]), unsafe_allow_html=True
                        )

                        # Display search entry point if available
                        if response["search_entry_point"]:
                            st.markdown(
                                format_message(
                                    response["search_entry_point"], is_hebrew=False
                                ),
                                unsafe_allow_html=True,
                            )

                    with context_tab:
                        # Wrap the context display in a container div for RTL support
                        context_html = f"""
                        <div dir="rtl" style="direction: rtl; text-align: right;">
                            {format_context_display(context)}
                        </div>
                        """
                        st.components.v1.html(context_html, height=400, scrolling=True)

                    with raw_tab:
                        # Wrap the raw context display in a container div for RTL support
                        raw_html = f"""
                        <div dir="rtl" style="direction: rtl; text-align: right;">
                            {format_raw_context(similar_convs)}
                        </div>
                        """
                        st.components.v1.html(raw_html, height=400, scrolling=True)

                    # Store in conversation history with context
                    st.session_state.conversation_history.append(
                        {"role": "user", "content": question}
                    )
                    st.session_state.conversation_history.append(
                        {
                            "role": "assistant",
                            "content": response["text"],
                            "context": context,
                        }
                    )


if __name__ == "__main__":
    main()
