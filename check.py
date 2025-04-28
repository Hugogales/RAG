# Requires transformers>=4.36.0
import torch.nn.functional as F
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModel,  pipeline
from typing import List
from tqdm import tqdm
import lancedb
import pandas as pd
import re
import gc
import json

torch.cuda.empty_cache()

def split_document(text: str, chunk_size: int = 1000, overlap: int = 100) -> List[str]:
    """
    Split a text into chunks suitable for RAG models, respecting sentence boundaries
    and including overlap between chunks.
    
    Args:
    text (str): The original text to split.
    chunk_size (int): Target size of each chunk in characters.
    overlap (int): Number of characters to overlap between chunks.
    
    Returns:
    List[str]: A list of text chunks.
    """
    # Split the text into sentences
    sentences = re.split('(?<=[.!?]) +', text)
    
    chunks = []
    current_chunk = ""
    
    for sentence in sentences:
        if len(current_chunk) + len(sentence) <= chunk_size:
            current_chunk += sentence + " "
        else:
            # If the current chunk is not empty, add it to the list of chunks
            if current_chunk:
                chunks.append(current_chunk.strip())
            
            # Start a new chunk, including some overlap from the previous chunk
            overlap_text = current_chunk[-overlap:] if len(current_chunk) > overlap else current_chunk
            current_chunk = overlap_text + sentence + " "
    
    # Add the last chunk if it's not empty
    if current_chunk:
        chunks.append(current_chunk.strip())
    
    return chunks

class embedding_model:
    def __init__(self, batch_size=64) -> None:
        model_path = 'Alibaba-NLP/gte-large-en-v1.5'
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModel.from_pretrained(
            model_path,
            trust_remote_code=True,
            attn_implementation='eager'  # Use eager implementation to avoid flash-attention warning
        )
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()
        self.batch_size = batch_size

    def tokenize_texts(self, texts: List[str]) -> torch.Tensor:
        all_embeddings = []
        for i in tqdm(range(0, len(texts), self.batch_size), desc="Processing batches"):
            batch_texts = texts[i:i+self.batch_size]
            batch_dict = self.tokenizer(
                batch_texts,
                max_length=512,  # Reduced from 8192
                padding=True, 
                truncation=True, 
                return_tensors='pt'
            )
            
            # Move batch to the same device as the model
            batch_dict = {k: v.to(self.device) for k, v in batch_dict.items()}
            
            with torch.no_grad():
                outputs = self.model(**batch_dict)
            embeddings = outputs.last_hidden_state[:, 0]
            all_embeddings.append(embeddings.cpu())  # Move back to CPU
        
            # Clear CUDA cache and garbage collect
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
        
        # Concatenate all embeddings
        embeddings = torch.cat(all_embeddings, dim=0)
        
        # Normalize embeddings
        embeddings = F.normalize(embeddings, p=2, dim=1)
        return embeddings

    def tokenize_text(self, text: str) -> torch.Tensor:
        return self.tokenize_texts([text])[0]

class LLM_model:
    def __init__(self) -> None:
        torch.manual_seed(0)
        model_id = "microsoft/Phi-3-medium-128k-instruct"
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.cuda_available = self.device.type == "cuda"
        
        # Load model with optimizations
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            trust_remote_code=True,
            device_map="auto",  # Automatically distribute model across available GPUs
        ).eval()
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        
        # Create pipeline without specifying device
        self.pipe = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            framework="pt",
        )
        
    @torch.inference_mode()
    def use(self, query: str, contexts: list[str]) -> str:
        context_string = "Context\n" + "\n".join(f"{i}: {context}" for i, context in enumerate(contexts))
            
        messages = [
            {"role": "system", "content": "You are an AI assistant specialized in answering questions based on provided contexts. Always refer to the given contexts to formulate your response. If the answer is not in the contexts, say so."},
            {"role": "user", "content": f"Here are the contexts to use for answering the upcoming question:\n\n{context_string}\n\nNow, please answer the following question using ONLY the information from these contexts. Question: {query}"},
            {"role": "assistant", "content": "Certainly! I'll answer the question based solely on the provided contexts. Let me analyze the information given:"},
            {"role": "user", "content": "Great! Now provide your answer, making sure to reference the specific context numbers you're using."}
        ]
        
        generation_args = {
            "max_new_tokens": 1000,
            "return_full_text": False,
            "temperature": 0.0,
            "do_sample": False,
            "num_beams": 1,
        }
        
        with torch.cuda.amp.autocast(enabled=self.cuda_available):
            output = self.pipe(messages, **generation_args)
            
        return output[0]['generated_text']

def get_texts():
    with open("articles_hugo.json", "r") as file:
        contents = json.load(file)
    
    chunks = []
    for article in contents:
        chunks += split_document(article)

    print(f"number of chunks: {len(chunks)}") 

    return chunks
    
debug = False
create_new_table = True

if __name__ == "__main__":

    # LLM model
    llm = LLM_model()

    # embedding model
    embedder = embedding_model()

    #database
    uri = "data/sample-lancedb"
    db = lancedb.connect(uri)

    if create_new_table:
        texts = get_texts()
        embeddings = embedder.tokenize_texts(texts)
        embeddings_np = embeddings.detach().numpy()
        # create table + database
        df = pd.DataFrame({'text': texts, 'vector': [embedding.tolist() for embedding in embeddings_np]})
        try:
            tbl = db.create_table("train_table", data=df)
        except Exception:
            db.drop_table("train_table")
            tbl = db.create_table("train_table", data=df)
    else:
        if debug:
            tbl = db.open_table("debug_table")
        else:
            tbl = db.open_table("train_table")

    # run loop
    print("Welcome to the RAG Model Testing Menu!")
    print("Type your query to get a response from the model.")
    print("Type 'STOP' or 'stop' or leave blank to exit.")
    print("-" * 50)

    while True:
        query = input("Enter your query: ").strip()
        if query.lower() == "stop" or query == "":
            print("Exiting the RAG Model Testing Menu. Goodbye!")
            break
        
        #use code
        embedding = embedder.tokenize_text(query)
        embedding = embedding.detach().numpy()
        response = tbl.search(embedding).limit(3).to_pandas()

        #similarity = F.cosine_similarity(torch.tensor(embedding).unsqueeze(0), torch.tensor(response['vector'].values[0]).unsqueeze(0)).item()
        print("-" * 50)
        output = llm.use(query=query, contexts=response["text"].tolist())

        print("-" * 50)
        print(f"Response: \n {output}")
        print("-" * 50)
        print("Context:")
        for i, context in enumerate(response["text"].tolist()):
            print(f"*** DOC {i} : \n {context} \n")
        print("-" * 50)
    
