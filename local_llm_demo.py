#!/usr/bin/env python3
"""
로컬 LLM (Ollama) 사용 RAG 데모
"""

import os
import logging
from dotenv import load_dotenv
from src.enhanced_rag_pipeline import EnhancedRAGPipeline
from src.llm_factory import LLMFactory, LLMConfig

logging.basicConfig(level=logging.INFO)
load_dotenv()

def check_ollama_status():
    """Ollama 서버 상태 확인"""
    print("🔍 Ollama 서버 상태 확인 중...")
    
    base_url = "http://localhost:11434"
    is_running = LLMFactory.check_ollama_server(base_url)
    
    if is_running:
        print("✅ Ollama 서버가 실행 중입니다.")
        
        # 사용 가능한 모델 목록
        models = LLMFactory.get_available_ollama_models(base_url)
        if models:
            print(f"📋 설치된 모델 ({len(models)}개):")
            for model in models:
                print(f"   - {model}")
        else:
            print("⚠️  설치된 모델이 없습니다.")
            return False
    else:
        print("❌ Ollama 서버에 연결할 수 없습니다.")
        print("   다음 명령어로 Ollama를 시작하세요:")
        print("   ollama serve")
        return False
    
    return True

def recommend_models():
    """추천 모델 표시"""
    print("\n🎯 추천 Ollama 모델:")
    print("=" * 50)
    
    recommendations = LLMConfig.RECOMMENDED_MODELS["ollama"]
    
    for model, info in recommendations.items():
        size = info.get("size", "Unknown")
        speed = info.get("speed", "Unknown")
        context = info.get("context", "Unknown")
        
        print(f"📦 {model}")
        print(f"   크기: {size}, 속도: {speed}, 컨텍스트: {context}")
        print(f"   설치: ollama pull {model}")
        print()

def interactive_model_setup():
    """대화형 모델 설정"""
    base_url = "http://localhost:11434"
    available_models = LLMFactory.get_available_ollama_models(base_url)
    
    if not available_models:
        print("❌ 사용 가능한 모델이 없습니다.")
        recommend_models()
        return None
    
    print(f"\n📋 사용 가능한 모델을 선택하세요:")
    for i, model in enumerate(available_models, 1):
        print(f"   {i}. {model}")
    
    try:
        choice = int(input(f"\n선택 (1-{len(available_models)}): "))
        if 1 <= choice <= len(available_models):
            selected_model = available_models[choice - 1].split(":")[0]  # 태그 제거
            print(f"✅ 선택된 모델: {selected_model}")
            return selected_model
        else:
            print("❌ 잘못된 선택입니다.")
            return None
    except ValueError:
        print("❌ 숫자를 입력해주세요.")
        return None

def demo_with_local_llm():
    """로컬 LLM 데모"""
    print("🦙 로컬 LLM (Ollama) RAG 데모")
    print("=" * 50)
    
    # Ollama 상태 확인
    if not check_ollama_status():
        return
    
    # 모델 선택
    selected_model = interactive_model_setup()
    if not selected_model:
        return
    
    # 임베딩 모델 선택
    print(f"\n🔢 임베딩 모델 선택:")
    print("   1. OpenAI (API 키 필요)")
    print("   2. SentenceTransformer (로컬)")
    
    embedding_choice = input("선택 (1-2): ").strip()
    
    if embedding_choice == "1":
        if not os.getenv("OPENAI_API_KEY"):
            print("❌ OpenAI API 키가 설정되지 않았습니다.")
            print("   SentenceTransformer를 사용합니다.")
            embedding_type = "sentence-transformer"
            embedding_model = "all-MiniLM-L6-v2"
        else:
            embedding_type = "openai"
            embedding_model = "text-embedding-3-small"
    else:
        embedding_type = "sentence-transformer"
        embedding_model = "all-MiniLM-L6-v2"
    
    print(f"📊 설정:")
    print(f"   LLM: Ollama/{selected_model}")
    print(f"   임베딩: {embedding_type}/{embedding_model}")
    
    # RAG 파이프라인 초기화
    try:
        print(f"\n🚀 RAG 파이프라인 초기화 중...")
        rag = EnhancedRAGPipeline(
            llm_type="ollama",
            llm_model=selected_model,
            embedding_type=embedding_type,
            embedding_model=embedding_model,
            collection_name="local_llm_docs"
        )
        print("✅ 초기화 완료!")
    except Exception as e:
        print(f"❌ 초기화 실패: {e}")
        return
    
    # 문서 로드
    medical_file = "data/medical_info.txt"
    if os.path.exists(medical_file):
        print(f"\n📄 의료 문서 로드 중...")
        result = rag.add_documents([medical_file])
        if result['success']:
            print(f"✅ 문서 로드 완료: {result['total_chunks']}개 청크")
        else:
            print(f"❌ 문서 로드 실패: {result['failed']}")
            return
    else:
        print(f"❌ 의료 문서를 찾을 수 없습니다: {medical_file}")
        return
    
    # 테스트 질문들
    test_questions = [
        "고혈압이 무엇인가요?",
        "당뇨병의 주요 증상은?", 
        "감기 예방법을 알려주세요",
        "응급상황 시 어떻게 해야 하나요?"
    ]
    
    print(f"\n🤖 테스트 질문 ({len(test_questions)}개):")
    print("=" * 50)
    
    for i, question in enumerate(test_questions, 1):
        print(f"\n[질문 {i}] {question}")
        print("-" * 30)
        
        try:
            result = rag.generate_answer(question, n_results=3)
            
            print(f"🩺 답변: {result['answer'][:200]}...")
            print(f"📚 출처: {', '.join(result.get('sources', []))}")
            print(f"🔧 사용 모델: {result.get('llm_info', 'Unknown')}")
            
        except Exception as e:
            print(f"❌ 오류: {e}")
    
    # 대화형 모드
    print(f"\n" + "=" * 50)
    print("💬 대화형 모드를 시작하시겠습니까?")
    user_choice = input("'y' 입력 시 시작: ").strip().lower()
    
    if user_choice in ['y', 'yes']:
        print(f"\n🎯 로컬 LLM 대화형 모드 시작!")
        print("종료: 'quit' 또는 'exit'")
        print("-" * 30)
        
        while True:
            try:
                question = input("\n👤 질문: ").strip()
                
                if question.lower() in ['quit', 'exit', '종료']:
                    print("👋 대화를 종료합니다!")
                    break
                
                if not question:
                    continue
                
                print("🤖 답변 생성 중...")
                result = rag.generate_answer(question)
                
                print(f"\n🦙 답변: {result['answer']}")
                if result.get('sources'):
                    print(f"📚 출처: {', '.join(result['sources'])}")
                
            except KeyboardInterrupt:
                print("\n👋 대화를 종료합니다!")
                break
            except Exception as e:
                print(f"❌ 오류: {e}")

def main():
    """메인 함수"""
    try:
        demo_with_local_llm()
    except KeyboardInterrupt:
        print("\n👋 프로그램을 종료합니다!")
    except Exception as e:
        print(f"❌ 예상치 못한 오류: {e}")

if __name__ == "__main__":
    main()