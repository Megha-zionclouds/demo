from flask import Flask, request, render_template, jsonify
import pandas as pd
from azure.search.documents import SearchClient
import matplotlib.pyplot as plt
from azure.storage.blob import BlobServiceClient
import os
import io
from openai import AzureOpenAI
from azure.core.credentials import AzureKeyCredential
from azure.ai.textanalytics import TextAnalyticsClient
import string
from azure.search.documents.models import VectorizedQuery

# Azure Blob Storage configuration
BLOB_CONNECTION_STRING = "DefaultEndpointsProtocol=https;AccountName=onestoragemegha;AccountKey=sbzFfN5fz0BbxfYoGNlK6ivO3zd8mKCdRIVFhKenCisZz0W3FHt8oCRRyw2MofY3y+D2ibUL9w0c+AStpqCjwA==;EndpointSuffix=core.windows.net"
BLOB_CONTAINER_NAME = "demo-csv"
AI_SERVICE_ENDPOINT="https://megha-demo-one.cognitiveservices.azure.com/"
AI_SERVICE_KEY="a49e29ee5d8b4909a20ed62cb7de1830"

 # Create client using endpoint and key
credential = AzureKeyCredential(AI_SERVICE_KEY)
ai_client = TextAnalyticsClient(endpoint=AI_SERVICE_ENDPOINT, credential=credential)

azure_oai_endpoint = "https://one-ml-megha.openai.azure.com/"
 # Initialize the Azure OpenAI client
azure_oai_key = "3f6dccba3ebd442797126c3a5fc0f982"
azure_oai_deployment = "gpt-35-turbo"
client = AzureOpenAI(
         azure_endpoint = azure_oai_endpoint, 
         api_key=azure_oai_key,  
         api_version="2024-02-15-preview"
         )
index_client = SearchClient(endpoint="https://meghademo.search.windows.net",
                            index_name="demo-1",
                            credential=AzureKeyCredential("tafEALGyafLmkt1KIOvfAyNx4uOycaWzY8h7zIfWLQAzSeArKSm3"))

app = Flask(__name__,static_folder='static')

# Initialize the BlobServiceClient
blob_service_client = BlobServiceClient.from_connection_string(BLOB_CONNECTION_STRING)

# Set up your Qdrant and model encoder
def get_embeddings(text):
        response = client.embeddings.create(
        model="text-embedding-3-large",  # Or any other embedding model
        input=text
    )
        return response.data[0].embedding

# Upload CSV to Azure Blob Storage
def upload_to_blob(file, filename):
    blob_client = blob_service_client.get_blob_client(container=BLOB_CONTAINER_NAME, blob=filename)
    blob_client.upload_blob(file)
    return blob_client.blob_name

# Retrieve CSV from Blob Storage
def download_from_blob(blob_name):
    container_client = blob_service_client.get_container_client(BLOB_CONTAINER_NAME)
    blob_client = container_client.get_blob_client(blob_name)
    blob_data = blob_client.download_blob()
    return pd.read_csv(io.BytesIO(blob_data.readall()))

# Route to render the home page
@app.route('/')
def index():
    return render_template('index.html')

# Route to upload CSV to Azure Blob Storage
@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    filename = file.filename
    # Upload to Azure Blob Storage
    blob_url = upload_to_blob(file, filename)
    index_file(blob_url)
    return jsonify({'message': 'File uploaded successfully!', 'blob_url': blob_url})

def index_file(blob_url):
    df = download_from_blob(blob_url)
    df['document_text'] = df.apply(lambda row: ' '.join(row.astype(str)), axis=1)
    

# Apply embedding generation for each row
    df['embeddings'] = df['document_text'].apply(lambda x: get_embeddings(x))
    documents = []
    for idx, row in df.iterrows():
        documents.append({
        "id": str(idx),
        "document_text": row['document_text'],
        "embeddings": row['embeddings']
        })

# Upload documents to the index
    index_client.upload_documents(documents=documents)

    return jsonify({'message': 'File indexed successfully!', 'blob_url': blob_url})

# Route to handle user query and generate results
@app.route('/ask', methods=['POST'])
def ask_question():
    user_input = request.json.get('question')

    # Download and load the CSV from Blob
    #df = download_from_blob(blob_url)
    # Create the Qdrant vectors
# Prepare documents for uploading
    # Search Qdrant with the user's query
    
    # Check if the user wants to quit
# Pure Vector Search

    query_vector = get_embeddings(user_input)
  
    vector_query = VectorizedQuery(vector=query_vector, k_nearest_neighbors=50, fields="embeddings")
# Use the below query to pass in the raw vector query instead of the query vectorization
# vector_query = RawVectorQuery(vector=generate_embeddings(query), k_nearest_neighbors=3, fields="vector")
  
    results = index_client.search(  
        search_text="",  
        vector_queries= [vector_query],
        top=5
        )  
  
    search_results = [result['document_text'] for result in results]

    #if not search_results:
     #   return jsonify({'answer': 'No valid information found'})

    # Prepare the results in a string format for the model
    search_results_str = "\n".join(search_results)
    print(search_results_str)

    # Modify the system prompt to instruct the model to strictly use search results
    prompt = f"Based on the following data:\n{search_results_str}\nYou must answer the user question creatively ONLY using this data. If the information is not sufficient, respond with 'No valid information found'. User question: {user_input}"

    # Here you can run any logic for generating graphs based on the result
    # If there's a need for graph, you can extract columns and generate plots

# Query the GPT model (Azure OpenAI)
    response = client.chat.completions.create(
            model=azure_oai_deployment,
            temperature=0.3,
            max_tokens=400,
            messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": user_input}
            ]
            )          
    generated_text = response.choices[0].message.content
    #print(type(generated_text))
 # Print the response
    #print("Response: " + generated_text + "\n")
    
 # Get key phrases

    return jsonify({'answer': generated_text})

if __name__ == '__main__':
    app.run(debug=True)



