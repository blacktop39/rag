import os
import logging
from typing import Any, Dict, Optional, Union
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_ollama import ChatOllama
from sentence_transformers import SentenceTransformer
import requests

logger = logging.getLogger(__name__)

class LLMFactory:
    """LLM 및 임베딩 모델 팩토리"""
    
    @staticmethod
    def create_llm(
        llm_type: str = "openai",
        model_name: str = "gpt-3.5-turbo",
        base_url: str = "http://localhost:11434",
        temperature: float = 0.1,
        **kwargs
    ) -> Any:
        """LLM 생성"""
        
        if llm_type.lower() == "openai":
            if not os.getenv("OPENAI_API_KEY"):
                raise ValueError("OpenAI를 사용하려면 OPENAI_API_KEY가 필요합니다.")
            
            return ChatOpenAI(
                model=model_name,
                temperature=temperature,
                **kwargs
            )
            
        elif llm_type.lower() == "ollama":
            # Ollama 서버 상태 확인
            if not LLMFactory.check_ollama_server(base_url):
                raise ConnectionError(f"Ollama 서버에 연결할 수 없습니다: {base_url}")
            
            # 모델이 다운로드되어 있는지 확인
            if not LLMFactory.check_ollama_model(model_name, base_url):
                logger.warning(f"모델 '{model_name}'이 Ollama에 없습니다. 자동으로 다운로드를 시도합니다.")
                LLMFactory.pull_ollama_model(model_name, base_url)
            
            return ChatOllama(
                model=model_name,
                base_url=base_url,
                temperature=temperature,
                **kwargs
            )
        
        else:
            raise ValueError(f"지원하지 않는 LLM 타입: {llm_type}")
    
    @staticmethod
    def create_embeddings(
        embedding_type: str = "openai",
        model_name: str = "text-embedding-3-small",
        **kwargs
    ) -> Any:
        """임베딩 모델 생성"""
        
        if embedding_type.lower() == "openai":
            if not os.getenv("OPENAI_API_KEY"):
                raise ValueError("OpenAI 임베딩을 사용하려면 OPENAI_API_KEY가 필요합니다.")
            
            return OpenAIEmbeddings(
                model=model_name,
                **kwargs
            )
            
        elif embedding_type.lower() == "sentence-transformer":
            return SentenceTransformerEmbeddings(
                model_name=model_name,
                **kwargs
            )
        
        else:
            raise ValueError(f"지원하지 않는 임베딩 타입: {embedding_type}")
    
    @staticmethod
    def check_ollama_server(base_url: str) -> bool:
        """Ollama 서버 상태 확인"""
        try:
            response = requests.get(f"{base_url}/api/tags", timeout=5)
            return response.status_code == 200
        except Exception as e:
            logger.error(f"Ollama 서버 연결 실패: {e}")
            return False
    
    @staticmethod
    def check_ollama_model(model_name: str, base_url: str) -> bool:
        """Ollama 모델 존재 확인"""
        try:
            response = requests.get(f"{base_url}/api/tags")
            if response.status_code == 200:
                models = response.json().get("models", [])
                model_names = [model["name"].split(":")[0] for model in models]
                return model_name in model_names
            return False
        except Exception as e:
            logger.error(f"Ollama 모델 확인 실패: {e}")
            return False
    
    @staticmethod
    def pull_ollama_model(model_name: str, base_url: str) -> bool:
        """Ollama 모델 다운로드"""
        try:
            logger.info(f"Ollama 모델 '{model_name}' 다운로드 중...")
            response = requests.post(
                f"{base_url}/api/pull",
                json={"name": model_name},
                timeout=300  # 5분 타임아웃
            )
            return response.status_code == 200
        except Exception as e:
            logger.error(f"Ollama 모델 다운로드 실패: {e}")
            return False
    
    @staticmethod
    def get_available_ollama_models(base_url: str) -> list:
        """사용 가능한 Ollama 모델 목록"""
        try:
            response = requests.get(f"{base_url}/api/tags")
            if response.status_code == 200:
                models = response.json().get("models", [])
                return [model["name"] for model in models]
            return []
        except Exception as e:
            logger.error(f"Ollama 모델 목록 조회 실패: {e}")
            return []

class SentenceTransformerEmbeddings:
    """SentenceTransformer 기반 임베딩 클래스"""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
    
    def embed_documents(self, texts: list) -> list:
        """문서 임베딩"""
        return self.model.encode(texts).tolist()
    
    def embed_query(self, text: str) -> list:
        """쿼리 임베딩"""
        return self.model.encode([text])[0].tolist()

class LLMConfig:
    """LLM 설정 관리"""
    
    # 추천 모델 설정
    RECOMMENDED_MODELS = {
        "openai": {
            "gpt-3.5-turbo": {"context": 4096, "speed": "fast", "cost": "low"},
            "gpt-4": {"context": 8192, "speed": "medium", "cost": "high"},
            "gpt-4-turbo": {"context": 128000, "speed": "medium", "cost": "medium"}
        },
        "ollama": {
            "llama3.2": {"context": 128000, "speed": "medium", "size": "3B"},
            "llama3.1": {"context": 128000, "speed": "medium", "size": "8B"},
            "qwen2.5": {"context": 32768, "speed": "fast", "size": "3B"},
            "gemma2": {"context": 8192, "speed": "fast", "size": "2B"},
            "phi3": {"context": 128000, "speed": "fast", "size": "3.8B"}
        }
    }
    
    RECOMMENDED_EMBEDDINGS = {
        "openai": {
            "text-embedding-3-small": {"dimensions": 1536, "cost": "low"},
            "text-embedding-3-large": {"dimensions": 3072, "cost": "medium"}
        },
        "sentence-transformer": {
            "all-MiniLM-L6-v2": {"dimensions": 384, "size": "small"},
            "all-mpnet-base-v2": {"dimensions": 768, "size": "medium"},
            "paraphrase-multilingual-MiniLM-L12-v2": {"dimensions": 384, "size": "small", "multilingual": True}
        }
    }
    
    @classmethod
    def get_recommended_models(cls, llm_type: str) -> Dict[str, Any]:
        """추천 모델 목록 반환"""
        return cls.RECOMMENDED_MODELS.get(llm_type, {})
    
    @classmethod
    def get_recommended_embeddings(cls, embedding_type: str) -> Dict[str, Any]:
        """추천 임베딩 모델 목록 반환"""
        return cls.RECOMMENDED_EMBEDDINGS.get(embedding_type, {})