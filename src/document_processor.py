import os
from typing import List, Dict, Any
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_openai import OpenAIEmbeddings
import logging
import hashlib

logger = logging.getLogger(__name__)

class DocumentProcessor:
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200, 
                 embedding_model: str = "text-embedding-3-small"):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
        )
        self.embeddings = OpenAIEmbeddings(model=embedding_model)
    
    def load_document(self, file_path: str) -> List[Dict[str, Any]]:
        try:
            file_extension = os.path.splitext(file_path)[1].lower()
            
            if file_extension == '.pdf':
                loader = PyPDFLoader(file_path)
            elif file_extension == '.txt':
                loader = TextLoader(file_path, encoding='utf-8')
            else:
                raise ValueError(f"Unsupported file type: {file_extension}")
            
            documents = loader.load()
            logger.info(f"Loaded {len(documents)} pages from {file_path}")
            return documents
            
        except Exception as e:
            logger.error(f"Error loading document {file_path}: {e}")
            raise
    
    def split_documents(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        try:
            split_docs = self.text_splitter.split_documents(documents)
            logger.info(f"Split into {len(split_docs)} chunks")
            return split_docs
        except Exception as e:
            logger.error(f"Error splitting documents: {e}")
            raise
    
    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        try:
            embeddings = self.embeddings.embed_documents(texts)
            logger.info(f"Generated embeddings for {len(texts)} texts")
            return embeddings
        except Exception as e:
            logger.error(f"Error generating embeddings: {e}")
            raise
    
    def generate_query_embedding(self, query: str) -> List[float]:
        try:
            embedding = self.embeddings.embed_query(query)
            return embedding
        except Exception as e:
            logger.error(f"Error generating query embedding: {e}")
            raise
    
    def process_file(self, file_path: str) -> Dict[str, Any]:
        try:
            documents = self.load_document(file_path)
            split_docs = self.split_documents(documents)
            
            texts = [doc.page_content for doc in split_docs]
            embeddings = self.generate_embeddings(texts)
            
            metadatas = []
            ids = []
            
            for i, doc in enumerate(split_docs):
                text_hash = hashlib.md5(doc.page_content.encode()).hexdigest()
                doc_id = f"{os.path.basename(file_path)}_{i}_{text_hash[:8]}"
                
                metadata = {
                    "source": file_path,
                    "chunk_index": i,
                    "total_chunks": len(split_docs),
                    "file_name": os.path.basename(file_path)
                }
                
                if hasattr(doc, 'metadata') and doc.metadata:
                    metadata.update(doc.metadata)
                
                metadatas.append(metadata)
                ids.append(doc_id)
            
            return {
                "texts": texts,
                "embeddings": embeddings,
                "metadatas": metadatas,
                "ids": ids
            }
            
        except Exception as e:
            logger.error(f"Error processing file {file_path}: {e}")
            raise