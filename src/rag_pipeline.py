import os
from typing import List, Dict, Any, Optional
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema import HumanMessage, SystemMessage
import logging
from dotenv import load_dotenv

from .document_processor import DocumentProcessor
from .vector_store import VectorStore

load_dotenv()

logger = logging.getLogger(__name__)

class RAGPipeline:
    def __init__(self, 
                 collection_name: str = "rag_documents",
                 persist_directory: str = "./chroma_db",
                 model_name: str = "gpt-3.5-turbo",
                 embedding_model: str = "text-embedding-3-small",
                 chunk_size: int = 1000,
                 chunk_overlap: int = 200):
        
        self.document_processor = DocumentProcessor(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            embedding_model=embedding_model
        )
        self.vector_store = VectorStore(
            collection_name=collection_name,
            persist_directory=persist_directory
        )
        self.llm = ChatOpenAI(model=model_name, temperature=0.1)
        
        self.system_prompt = """당신은 질문-답변 AI 어시스턴트입니다. 제공된 컨텍스트를 바탕으로 정확하고 도움이 되는 답변을 제공하세요.

컨텍스트 정보:
{context}

다음 규칙을 따르세요:
1. 컨텍스트에 없는 정보는 추측하지 마세요
2. 답변할 수 없는 경우 솔직히 모른다고 하세요
3. 가능한 한 구체적이고 정확한 답변을 제공하세요
4. 출처를 언급할 때는 파일명을 포함하세요"""

    def add_documents(self, file_paths: List[str]) -> Dict[str, Any]:
        results = {
            "success": [],
            "failed": [],
            "total_chunks": 0
        }
        
        for file_path in file_paths:
            try:
                if not os.path.exists(file_path):
                    results["failed"].append({"file": file_path, "error": "File not found"})
                    continue
                
                processed_data = self.document_processor.process_file(file_path)
                
                self.vector_store.add_documents(
                    documents=processed_data["texts"],
                    embeddings=processed_data["embeddings"],
                    metadatas=processed_data["metadatas"],
                    ids=processed_data["ids"]
                )
                
                results["success"].append({
                    "file": file_path,
                    "chunks": len(processed_data["texts"])
                })
                results["total_chunks"] += len(processed_data["texts"])
                
                logger.info(f"Successfully added {file_path} with {len(processed_data['texts'])} chunks")
                
            except Exception as e:
                error_msg = f"Error processing {file_path}: {str(e)}"
                logger.error(error_msg)
                results["failed"].append({"file": file_path, "error": str(e)})
        
        return results
    
    def search_documents(self, query: str, n_results: int = 5) -> List[Dict[str, Any]]:
        try:
            query_embedding = self.document_processor.generate_query_embedding(query)
            search_results = self.vector_store.search(query_embedding, n_results)
            
            formatted_results = []
            if search_results["documents"] and search_results["documents"][0]:
                for i, (doc, metadata, distance) in enumerate(zip(
                    search_results["documents"][0],
                    search_results["metadatas"][0],
                    search_results["distances"][0]
                )):
                    formatted_results.append({
                        "content": doc,
                        "metadata": metadata,
                        "similarity_score": 1 - distance,
                        "rank": i + 1
                    })
            
            return formatted_results
            
        except Exception as e:
            logger.error(f"Error searching documents: {e}")
            raise
    
    def generate_answer(self, query: str, n_results: int = 5) -> Dict[str, Any]:
        try:
            search_results = self.search_documents(query, n_results)
            
            if not search_results:
                return {
                    "answer": "죄송합니다. 관련된 정보를 찾을 수 없습니다.",
                    "sources": [],
                    "query": query
                }
            
            context = "\n\n".join([
                f"[출처: {result['metadata'].get('file_name', 'Unknown')}]\n{result['content']}"
                for result in search_results
            ])
            
            prompt = ChatPromptTemplate.from_messages([
                ("system", self.system_prompt),
                ("human", "질문: {question}")
            ])
            
            chain = prompt | self.llm
            
            response = chain.invoke({
                "context": context,
                "question": query
            })
            
            sources = list(set([
                result['metadata'].get('file_name', 'Unknown')
                for result in search_results
            ]))
            
            return {
                "answer": response.content,
                "sources": sources,
                "query": query,
                "search_results": search_results
            }
            
        except Exception as e:
            logger.error(f"Error generating answer: {e}")
            return {
                "answer": f"답변 생성 중 오류가 발생했습니다: {str(e)}",
                "sources": [],
                "query": query
            }
    
    def get_stats(self) -> Dict[str, Any]:
        collection_info = self.vector_store.get_collection_info()
        return {
            "collection_name": collection_info["name"],
            "total_documents": collection_info["count"],
            "persist_directory": collection_info["persist_directory"]
        }