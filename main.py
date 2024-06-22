from groq import Groq
import streamlit as st
import fitz
from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer
import os
from dotenv import load_dotenv

load_dotenv()
GROQ_API_KEY = os.environ["GROQ_API_KEY"]
PINECONE_API_KEY = os.environ["PINECONE_API_KEY"]

# create embedding model (dimension 768)
embedded_model = SentenceTransformer("all-mpnet-base-v2")

# embedding


def embedding_text(pdf_text):
    sentences = pdf_text.split('.')
    embeddings = embedded_model.encode(sentences)
    return sentences, embeddings

# get user input embedding


def user_input_embedded_text(user_input):
    query_embedding = embedded_model.encode([user_input])
    return query_embedding


# pinecone api key (key name - chatbot)
pc = Pinecone(api_key=PINECONE_API_KEY)

#  connect with pinecone
try:
    index = pc.Index("quickstart")
    print(index.describe_index_stats())
except Exception as e:
    st.error(f"Error connecting to Pinecone: {e}")


# add data to the vector db


def upsert_embeddings(embeddings, sentence):
    try:
        vectors = []
        for i, (embedding, sentence) in enumerate(zip(embeddings, sentence)):
            vectors.append({
                "id": str(i),
                "values": embedding.tolist(),
                "metadata": {"text": sentence}})
        index.upsert(vectors)
    except Exception as e:
        st.error(f"Error in upserting: {e}")


# save the question and answer in text file (set the new content to the top)

def save_data(question, response):
    try:
        with open('history.txt', 'r', encoding='utf-8') as file:
            previous_data = file.read()
    except Exception as e:
        st.error(f"File Not Found: {e}")
        previous_data = " "

    new_data = f"\nYou:\n\n{question}\n\nBot:\n\n{response}\n\n{'-'*50}\n\n"
    try:
        with open('history.txt', 'w', encoding='utf-8') as file:
            file.write(f"{new_data}")
            file.write(f"{previous_data}")
    except Exception as e:
        st.error(f"File Could not be write: {e}")


# display history

def display_history():
    try:
        with open('history.txt', 'r', encoding='utf-8') as file:
            history_content = file.read()
            st.markdown(f"<div style='height: 400px; overflow-y: scroll;overflow-x: scroll; padding: 10px; border: 1px solid #ccc; border-radius: 5px;'>{
                        history_content}</div>", unsafe_allow_html=True)
    except FileNotFoundError as e:
        st.error(f"File Not Found: {e}")
    except Exception as e:
        st.error(f"Error displaying history: {e}")


# extract the details in the pdf

def pdf_to_text(uploaded_pdf):
    if uploaded_pdf is not None:
        try:
            with fitz.open(stream=uploaded_pdf.read(), filetype='pdf') as pdf:
                pdf_text = ''
                for i in range(len(pdf)):
                    page = pdf.load_page(i)
                    pdf_text += page.get_text()
            return pdf_text
        except Exception as e:
            st.error(f"Error reading PDF: {e}")
            return None


# get summarize


def get_summarize(chat_history):
    try:
        if chat_history != "":
            client = Groq(
                api_key=(GROQ_API_KEY),
            )
            chat_completion = client.chat.completions.create(
                messages=[
                    {
                        "role": "user",
                        "content": f"make this summarize and make the output much as using less words.dont exceed 1000 words. and dont mention this is summarization {chat_history}",
                    }
                ],
                model="llama3-8b-8192",
            )
            get_summarize_text = chat_completion.choices[0].message.content
            return get_summarize_text
        else:
            return chat_history
    except Exception as e:
        st.error(f"Unexpected Error: {e}")


def main():
    st.title("Chat Bot")

    # Get user input
    user_input = st.text_input("Enter the Question here")
    # Get pdf
    uploaded_pdf = st.file_uploader("choose a file", type=['pdf'])

    pdf_text = None

    # st.write(user_input)

    try:

        # embedding the user input
        if user_input:
            embedded_query = user_input_embedded_text(user_input)
            output = index.query(vector=embedded_query[0].tolist(
            ), top_k=5, include_values=False, include_metadata=True)
            rag_response = " "
            for i in output['matches']:
                rag_response += i['metadata']['text']
                # st.write(i['metadata']['text'])

        # load chat history
        try:
            with open('history.txt', 'r', encoding='utf-8') as history:
                chat_history = history.read()
                summarized_chat_history = get_summarize(chat_history)
        except FileNotFoundError as e:
            st.error(f"File Not Found: {e}")

        # request
        if user_input:

            # Create the request
            client = Groq(
                api_key=GROQ_API_KEY
            )

            chat_completion = client.chat.completions.create(
                messages=[
                    {
                        "role": "user",
                        "content": f"Question: {user_input} \n History chat: {summarized_chat_history} \n New Updates: {rag_response} "
                    }
                ],
                model="llama3-8b-8192",
            )

            response = chat_completion.choices[0].message.content

            # Save the content
            save_data(user_input, response)

        # Display history
        display_history()

        # convert pdf to text
        if uploaded_pdf:
            pdf_text = pdf_to_text(uploaded_pdf)

        # embedding the text
        if pdf_text:
            sentence, embeddings = embedding_text(pdf_text)

            # add the embedded text to the vector db
            upsert_embeddings(embeddings, sentence)
            # st.write(embeddings)
            # st.write(sentence)

        c1, c2, c3, c4, c5, c6, c7, c8 = st.columns(
            [2, 2, 1, 1, 1, 1, 1, 1])

        # clear chat history
        try:
            with c1:
                if st.button("Clear History"):
                    with open('history.txt', 'w', encoding='utf-8') as file:
                        file.write("")
        except Exception as e:
            st.error(f"Error in cleaning Chat History:{e}")

        # clear vector db
        try:
            with c2:
                if st.button("Clear RAG"):
                    index.delete(delete_all=True)
        except Exception as e:
            st.error(f"Error in cleaning RAG: {e} ")

    except Exception as e:
        st.error(f"An Exception occur: {e} ")


if __name__ == "__main__":
    main()
