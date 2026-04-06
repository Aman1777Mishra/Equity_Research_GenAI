import streamlit as st
import web_scraper
import Chunk_and_Embed
import LLMChain



url = st.sidebar.text_input("Enter your URL")
st.header("Equity News Article Research")

#process_url = st.button("Process")

main_placeholder = st.empty()
question = main_placeholder.text_input("Enter your question here")
process = st.button("Answer")
main_placeholder_2 = st.empty()

if process:
    main_placeholder_2.text("Web Scraping...")
    document = web_scraper.web_scrape(url)

    main_placeholder_2.text("Chunking and Embedding...")
    vector_db_index = Chunk_and_Embed.chunk_and_embed(document)

    main_placeholder_2.text("LLM Thinking...")
    results = LLMChain.llmchain(vector_db_index, question)
    main_placeholder_2.text(results) 
