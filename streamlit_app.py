import os
import streamlit as st
from openai import OpenAI

# Optional import for Google Gemini (genai). We import lazily later if needed.

# Show title and description.
st.title("💬 Chatbot")
st.write(
    "This is a simple chatbot that can use OpenAI (GPT) or Google Gemini (via the google-genai client)."
)

# Choose provider: OpenAI or Gemini
provider = st.radio("Model provider", ["OpenAI", "Gemini (Google)"], index=0)

if provider == "OpenAI":
    # Ask user for their OpenAI API key via `st.text_input`.
    openai_api_key = st.text_input("OpenAI API Key", type="password")
    if not openai_api_key:
        st.info("Please add your OpenAI API key to continue.", icon="🗝️")
    else:

        # Create an OpenAI client.
        client = OpenAI(api_key=openai_api_key)

        # Create a session state variable to store the chat messages. This ensures that the
        # messages persist across reruns.
        if "messages" not in st.session_state:
            st.session_state.messages = []

    # Display the existing chat messages via `st.chat_message`.
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Create a chat input field to allow the user to enter a message. This will display
    # automatically at the bottom of the page.
    if prompt := st.chat_input("What is up?"):

        # Store and display the current prompt.
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate a response using the OpenAI API (streaming supported by the client)
        stream = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": m["role"], "content": m["content"]} for m in st.session_state.messages],
            stream=True,
        )

        # Stream the response to the chat using `st.write_stream`, then store it in
        # session state.
        with st.chat_message("assistant"):
            response = st.write_stream(stream)
        st.session_state.messages.append({"role": "assistant", "content": response})

else:
    # Gemini (Google) path
    st.markdown("Enter your Google Gemini API key in the field below or set the environment variable `GEMINI_API_KEY`.")
    gemini_api_key = st.text_input("Gemini (GEMINI_API_KEY)", type="password")
    # Prefer explicit input, fall back to environment variable
    if not gemini_api_key:
        gemini_api_key = os.environ.get("GEMINI_API_KEY", "")

    if not gemini_api_key:
        st.info("Please add your Gemini API key to continue.", icon="🗝️")
    else:
        # Lazy import of genai client
        try:
            from google import genai
        except Exception as e:
            st.error("The google-genai package is not installed or failed to import. Make sure it's in requirements.txt and installed.")
            st.stop()

        # Create a genai client which will read key from the environment or accept a key
        # Note: the genai client reads credentials from the environment variable `GEMINI_API_KEY`.
        # We'll set it in the environment for the duration of this run if provided via text input.
        os.environ["GEMINI_API_KEY"] = gemini_api_key
        client = genai.Client()

        # Session state for messages
        if "messages" not in st.session_state:
            st.session_state.messages = []

        # Display previous messages
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        if prompt := st.chat_input("What is up?"):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            # Use the genai client to generate content. This is a simple non-streaming
            # call — adjust to streaming if needed and supported by your genai client version.
            response = client.models.generate_content(model="gemini-2.5-flash", contents=prompt)

            # Best-effort extraction of human-readable text from the genai response.
            # Different versions of google-genai expose different shapes; prefer
            # `response.candidates[0].content.parts[*].text` when available.
            text = ""
            try:
                # Newer response shape: response.candidates -> Candidate.content.parts -> Part.text
                if hasattr(response, "candidates") and response.candidates:
                    candidate = response.candidates[0]
                    content = getattr(candidate, "content", None)
                    parts = getattr(content, "parts", None)
                    if parts:
                        # Join all part.text values
                        text = "".join(getattr(p, "text", str(p)) for p in parts).strip()
                    else:
                        # candidate.content may be a plain string or object
                        text = str(content).strip()
                # Fallbacks
                elif hasattr(response, "text") and response.text:
                    text = str(response.text).strip()
                elif hasattr(response, "content") and response.content:
                    text = str(response.content).strip()
                else:
                    text = str(response).strip()
            except Exception:
                # Last resort: string representation
                text = str(response).strip()

            # Make the assistant reply look clean in the UI
            if not text:
                text = "(no text returned from Gemini)"

            with st.chat_message("assistant"):
                st.markdown(text)
            st.session_state.messages.append({"role": "assistant", "content": text})
