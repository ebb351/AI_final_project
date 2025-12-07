"""
Gradio Chat Interface for Erica AI Tutor.

Provides a user-friendly chat UI that connects to the GraphRAG API.
"""
import gradio as gr
import httpx
import logging

logger = logging.getLogger(__name__)


def create_chat_interface(api_base_url: str = "http://localhost:8000"):
    """
    Create Gradio chat interface for Erica.

    Args:
        api_base_url: Base URL of the GraphRAG API

    Returns:
        Gradio Blocks demo
    """

    # Define LaTeX delimiters for math rendering
    latex_delimiters = [
        {"left": "$$", "right": "$$", "display": True},
        {"left": "$", "right": "$", "display": False},
        {"left": "\\[", "right": "\\]", "display": True},
        {"left": "\\(", "right": "\\)", "display": False},
    ]

    # Create chat interface with LaTeX support using Blocks
    with gr.Blocks(title="Erica - AI Course Tutor") as demo:
        gr.Markdown("""
        # 🎓 Erica - AI Course Tutor

        Ask questions about AI and Machine Learning concepts from the Introduction to AI course.

        Erica uses a Knowledge Graph with 16,000+ entities to provide accurate, sourced answers.

        **Note:** Responses take 3-5 minutes on CPU as the LLM generates detailed explanations.
        """)

        chatbot = gr.Chatbot(
            label="Chat",
            height=500,
            latex_delimiters=latex_delimiters,
        )

        msg = gr.Textbox(
            label="Your question",
            placeholder="Ask about AI/ML concepts...",
            scale=4,
        )

        with gr.Row():
            submit_btn = gr.Button("Submit", variant="primary")
            clear_btn = gr.Button("Clear")

        gr.Examples(
            examples=[
                "What is attention in transformers and can you provide a Python example?",
                "What is CLIP and how is it used in computer vision applications?",
                "Can you explain the variational lower bound and how it relates to Jensen's inequality?",
                "How does backpropagation work in neural networks?",
                "What is the difference between Q-learning and policy gradient methods?",
            ],
            inputs=msg,
        )

        def user_submit(message, history):
            """Add user message to history using messages format."""
            if not message.strip():
                return "", history
            # Gradio 6.0 messages format: list of dicts with role and content
            history = history + [{"role": "user", "content": message}]
            return "", history

        def bot_respond(history):
            """Generate bot response using messages format."""
            if not history:
                return history

            # Get the last user message - handle Gradio 6.0 content format
            last_content = history[-1]["content"]
            # Content can be a string or a list of content parts
            if isinstance(last_content, list):
                # Extract text from content parts
                user_message = "".join(
                    part.get("text", "") if isinstance(part, dict) else str(part)
                    for part in last_content
                )
            else:
                user_message = last_content

            # Add loading message
            history = history + [{"role": "assistant", "content": "🔍 Searching knowledge graph and generating response...\n\n*This may take 3-5 minutes on CPU.*"}]
            yield history

            try:
                with httpx.Client(timeout=600) as client:
                    response = client.post(
                        f"{api_base_url}/query",
                        json={"question": user_message}
                    )
                    response.raise_for_status()
                    data = response.json()

                    answer = data.get("answer", "No response generated.")
                    citations = data.get("citations", [])
                    entities = data.get("entities_used", [])
                    latency_ms = data.get("latency_ms", 0)

                    # Format response with citations
                    formatted = answer

                    # Add citations section if available (deduplicated by URL)
                    if citations:
                        formatted += "\n\n---\n**📚 Sources:**\n"
                        seen_urls = set()
                        for cite in citations:
                            title = cite.get('title', 'Unknown')
                            url = cite.get('url', '')
                            # Skip duplicates based on URL (or title if no URL)
                            dedup_key = url if url else title
                            if dedup_key in seen_urls:
                                continue
                            seen_urls.add(dedup_key)
                            if url:
                                formatted += f"- [{title}]({url})\n"
                            else:
                                formatted += f"- {title}\n"

                    # Add entities used
                    if entities:
                        formatted += f"\n**🔗 Concepts used:** {', '.join(entities)}"

                    # Add timing info
                    if latency_ms:
                        seconds = latency_ms / 1000
                        formatted += f"\n\n*Response time: {seconds:.1f}s*"

                    # Update the assistant message with actual response
                    history[-1]["content"] = formatted

            except httpx.TimeoutException:
                history[-1]["content"] = "⏱️ **Request timed out.** The server is taking too long to respond. Please try again or simplify your question."
            except httpx.ConnectError:
                history[-1]["content"] = "❌ **Connection error.** Could not connect to the API server. Please ensure the service is running."
            except httpx.HTTPStatusError as e:
                history[-1]["content"] = f"❌ **API error ({e.response.status_code}):** {e.response.text}"
            except Exception as e:
                logger.error(f"Chat error: {e}")
                history[-1]["content"] = f"❌ **Error:** {str(e)}"

            yield history

        # Wire up events
        msg.submit(user_submit, [msg, chatbot], [msg, chatbot], queue=False).then(
            bot_respond, chatbot, chatbot
        )
        submit_btn.click(user_submit, [msg, chatbot], [msg, chatbot], queue=False).then(
            bot_respond, chatbot, chatbot
        )
        clear_btn.click(lambda: ("", []), None, [msg, chatbot], queue=False)

    return demo


def create_standalone_app(api_base_url: str = "http://localhost:8000"):
    """
    Create a standalone Gradio app (not mounted in FastAPI).

    Useful for development/testing.
    """
    demo = create_chat_interface(api_base_url)
    return demo


if __name__ == "__main__":
    # Run standalone for testing
    demo = create_standalone_app()
    demo.launch(server_name="0.0.0.0", server_port=7860)
