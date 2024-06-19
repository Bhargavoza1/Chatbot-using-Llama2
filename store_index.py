from src.helper import load_pdf, text_split, download_hugging_face_embeddings
from langchain.vectorstores import Pinecone
import pinecone
from dotenv import load_dotenv
import os
from langchain_pinecone import PineconeVectorStore

load_dotenv()



index_name = "medical-chatbot"

# print(PINECONE_API_KEY)
# print(PINECONE_API_ENV)

try:
    # Get the current working directory (where the script is executed)
    current_working_directory = os.getcwd()

    data_directory = os.path.join(current_working_directory, "data")

    extracted_data = load_pdf(data_directory)
    print("PDFs loaded successfully.")
except FileNotFoundError as e:
    print(e)

text_chunks = text_split(extracted_data)
embeddings = download_hugging_face_embeddings()


#Creating Embeddings for Each of The Text Chunks & storing
docsearch=PineconeVectorStore.from_texts([t.page_content for t in text_chunks], embeddings, index_name=index_name)
