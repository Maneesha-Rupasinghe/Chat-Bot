# Chat-Bot

This project implements a Chat Bot using Streamlit, Pinecone, Groq, and Sentence Transformers to handle user queries, PDF uploads for text extraction, and summarization. It includes error handling and maintains chat history locally

## Get started

1. Install libraries

   ```bash
   pip install streamlit
   ```

   ```bash
   pip install pymupdf
   ```

   ```bash
   pip install pinecone-client
   ```

   ```bash
   pip install groq
   ```

   ```bash
   pip install sentence-transformers
   ```
2. Download and install wkhtmltopdf
   ```
   download link - https://wkhtmltopdf.org/downloads.html
   ```
   ```
   save the path in the system variables
   ```
   ```
   ##(add that path in the code also)
   ```
   
4. Start the app

   ```bash
    streamlit run main.py
   ```

## NOTE: Please make sure to replace your API keys 
(GROQ_API_KEY and PINECONE_API_KEY)
