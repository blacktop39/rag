import os
from typing import List, Dict, Any, Optional
from langchain.prompts import ChatPromptTemplate
import logging
from dotenv import load_dotenv

from .llm_factory import LLMFactory, LLMConfig
from .vector_store import VectorStore

load_dotenv()
logger = logging.getLogger(__name__)

class EnhancedRAGPipeline:
    """로컬 LLM 지원이 포함된 향상된 RAG 파이프라인"""
    
    def __init__(self, 
                 collection_name: str = "rag_documents",
                 persist_directory: str = "./chroma_db",
                 llm_type: str = "openai",
                 llm_model: str = "gpt-3.5-turbo",
                 embedding_type: str = "openai", 
                 embedding_model: str = "text-embedding-3-small",
                 ollama_base_url: str = "http://localhost:11434",
                 chunk_size: int = 1000,
                 chunk_overlap: int = 200):
        
        self.llm_type = llm_type
        self.llm_model = llm_model
        self.embedding_type = embedding_type
        self.embedding_model = embedding_model
        self.ollama_base_url = ollama_base_url
        
        # LLM 초기화
        try:
            self.llm = LLMFactory.create_llm(
                llm_type=llm_type,
                model_name=llm_model,
                base_url=ollama_base_url,
                temperature=0.1
            )
            logger.info(f"LLM 초기화 완료: {llm_type}/{llm_model}")
        except Exception as e:
            logger.error(f"LLM 초기화 실패: {e}")
            raise
        
        # 임베딩 모델 초기화
        try:
            self.embeddings = LLMFactory.create_embeddings(
                embedding_type=embedding_type,
                model_name=embedding_model
            )
            logger.info(f"임베딩 모델 초기화 완료: {embedding_type}/{embedding_model}")
        except Exception as e:
            logger.error(f"임베딩 모델 초기화 실패: {e}")
            raise
        
        # 벡터 저장소 초기화
        self.vector_store = VectorStore(
            collection_name=collection_name,
            persist_directory=persist_directory
        )
        
        # 문서 처리용 텍스트 분할기
        from langchain.text_splitter import RecursiveCharacterTextSplitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
        )
        
        # 기본 시스템 프롬프트
        self.system_prompt = """당신은 질문-답변 AI 어시스턴트입니다. 제공된 컨텍스트를 바탕으로 정확하고 도움이 되는 답변을 제공하세요.

컨텍스트 정보:
{context}

다음 규칙을 따르세요:
1. 컨텍스트에 없는 정보는 추측하지 마세요
2. 답변할 수 없는 경우 솔직히 모른다고 하세요
3. 가능한 한 구체적이고 정확한 답변을 제공하세요
4. 출처를 언급할 때는 파일명을 포함하세요"""

    def load_document(self, file_path: str) -> List[Dict[str, Any]]:
        """문서 로드"""
        try:
            from langchain_community.document_loaders import PyPDFLoader, TextLoader
            
            file_extension = os.path.splitext(file_path)[1].lower()
            
            if file_extension == '.pdf':
                loader = PyPDFLoader(file_path)
            elif file_extension == '.txt':
                loader = TextLoader(file_path, encoding='utf-8')
            else:
                raise ValueError(f"지원하지 않는 파일 형식: {file_extension}")
            
            documents = loader.load()
            logger.info(f"문서 로드 완료: {file_path} ({len(documents)} 페이지)")
            return documents
            
        except Exception as e:
            logger.error(f"문서 로드 실패 {file_path}: {e}")
            raise

    def process_documents(self, documents: List[Dict[str, Any]]) -> Dict[str, Any]:
        """문서 처리 (분할 및 임베딩)"""
        try:
            # 텍스트 분할
            split_docs = self.text_splitter.split_documents(documents)
            logger.info(f"문서 분할 완료: {len(split_docs)} 청크")
            
            # 텍스트 추출
            texts = [doc.page_content for doc in split_docs]
            
            # 임베딩 생성
            if self.embedding_type == "sentence-transformer":
                embeddings = self.embeddings.embed_documents(texts)
            else:
                embeddings = self.embeddings.embed_documents(texts)
            
            logger.info(f"임베딩 생성 완료: {len(embeddings)} 벡터")
            
            # 메타데이터 및 ID 생성
            import hashlib
            metadatas = []
            ids = []
            
            for i, doc in enumerate(split_docs):
                text_hash = hashlib.md5(doc.page_content.encode()).hexdigest()
                doc_id = f"doc_{i}_{text_hash[:8]}"
                
                metadata = {
                    "chunk_index": i,
                    "total_chunks": len(split_docs)
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
            logger.error(f"문서 처리 실패: {e}")
            raise

    def add_documents(self, file_paths: List[str]) -> Dict[str, Any]:
        """문서들을 벡터 데이터베이스에 추가"""
        results = {
            "success": [],
            "failed": [],
            "total_chunks": 0
        }
        
        for file_path in file_paths:
            try:
                if not os.path.exists(file_path):
                    results["failed"].append({"file": file_path, "error": "파일을 찾을 수 없습니다"})
                    continue
                
                # 문서 로드 및 처리
                documents = self.load_document(file_path)
                processed_data = self.process_documents(documents)
                
                # 벡터 저장소에 추가
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
                
                logger.info(f"문서 추가 완료: {file_path} ({len(processed_data['texts'])} 청크)")
                
            except Exception as e:
                error_msg = f"문서 처리 실패: {str(e)}"
                logger.error(f"{file_path}: {error_msg}")
                results["failed"].append({"file": file_path, "error": str(e)})
        
        return results

    def search_documents(self, query: str, n_results: int = 5) -> List[Dict[str, Any]]:
        """문서 검색"""
        try:
            # 쿼리 임베딩 생성
            if self.embedding_type == "sentence-transformer":
                query_embedding = self.embeddings.embed_query(query)
            else:
                query_embedding = self.embeddings.embed_query(query)
            
            # 벡터 검색
            search_results = self.vector_store.search(query_embedding, n_results)
            
            # 결과 포맷팅
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
            logger.error(f"문서 검색 실패: {e}")
            raise

    def generate_answer(self, query: str, n_results: int = 5, custom_prompt: Optional[str] = None) -> Dict[str, Any]:
        """질문에 대한 답변 생성"""
        try:
            # 문서 검색
            search_results = self.search_documents(query, n_results)
            
            if not search_results:
                return {
                    "answer": "죄송합니다. 관련된 정보를 찾을 수 없습니다.",
                    "sources": [],
                    "query": query,
                    "llm_info": f"{self.llm_type}/{self.llm_model}"
                }
            
            # 컨텍스트 구성
            context = "\n\n".join([
                f"[출처: {result['metadata'].get('source', 'Unknown')}]\n{result['content']}"
                for result in search_results
            ])
            
            # 프롬프트 구성
            system_prompt = custom_prompt if custom_prompt else self.system_prompt
            prompt = ChatPromptTemplate.from_messages([
                ("system", system_prompt),
                ("human", "질문: {question}")
            ])
            
            # LLM 체인 실행
            chain = prompt | self.llm
            
            response = chain.invoke({
                "context": context,
                "question": query
            })
            
            # 출처 정보
            sources = list(set([
                result['metadata'].get('source', 'Unknown')
                for result in search_results
            ]))
            
            return {
                "answer": response.content,
                "sources": sources,
                "query": query,
                "search_results": search_results,
                "llm_info": f"{self.llm_type}/{self.llm_model}",
                "embedding_info": f"{self.embedding_type}/{self.embedding_model}"
            }
            
        except Exception as e:
            logger.error(f"답변 생성 실패: {e}")
            return {
                "answer": f"답변 생성 중 오류가 발생했습니다: {str(e)}",
                "sources": [],
                "query": query,
                "llm_info": f"{self.llm_type}/{self.llm_model}"
            }

    def get_stats(self) -> Dict[str, Any]:
        """시스템 상태 정보"""
        collection_info = self.vector_store.get_collection_info()
        
        stats = {
            "collection_name": collection_info["name"],
            "total_documents": collection_info["count"],
            "persist_directory": collection_info["persist_directory"],
            "llm_type": self.llm_type,
            "llm_model": self.llm_model,
            "embedding_type": self.embedding_type,
            "embedding_model": self.embedding_model
        }
        
        # Ollama 관련 정보 추가
        if self.llm_type == "ollama":
            stats["ollama_base_url"] = self.ollama_base_url
            stats["ollama_available"] = LLMFactory.check_ollama_server(self.ollama_base_url)
            if stats["ollama_available"]:
                stats["ollama_models"] = LLMFactory.get_available_ollama_models(self.ollama_base_url)
        
        return stats

    @classmethod
    def get_recommended_configs(cls) -> Dict[str, Any]:
        """추천 설정 반환"""
        return {
            "llm_models": LLMConfig.RECOMMENDED_MODELS,
            "embedding_models": LLMConfig.RECOMMENDED_EMBEDDINGS
        }