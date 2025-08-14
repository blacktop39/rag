import os
import logging
from dotenv import load_dotenv
from src.rag_pipeline import RAGPipeline

logging.basicConfig(level=logging.INFO)
load_dotenv()

def create_sample_document() -> str:
    sample_content = """
인공지능과 머신러닝

인공지능(AI)은 인간의 지능을 모방하는 컴퓨터 시스템을 의미합니다. 
머신러닝은 AI의 하위 분야로, 데이터로부터 패턴을 학습하는 기법입니다.

주요 머신러닝 알고리즘:
1. 선형 회귀: 연속적인 값을 예측하는 알고리즘
2. 로지스틱 회귀: 분류 문제를 해결하는 알고리즘  
3. 의사결정 트리: 규칙 기반의 분류/회귀 알고리즘
4. 랜덤 포레스트: 여러 의사결정 트리를 조합한 앙상블 방법
5. 신경망: 인간의 뇌 구조를 모방한 알고리즘

딥러닝은 여러 층의 신경망을 사용하는 머신러닝의 한 분야입니다.
주요 딥러닝 모델로는 CNN(합성곱 신경망), RNN(순환 신경망), 
Transformer 등이 있습니다.

RAG(Retrieval-Augmented Generation)는 정보 검색과 텍스트 생성을 
결합한 AI 기법으로, 외부 지식을 활용하여 더 정확한 답변을 생성합니다.
"""
    
    os.makedirs("data", exist_ok=True)
    with open("data/ai_guide.txt", "w", encoding="utf-8") as f:
        f.write(sample_content)
    
    return "data/ai_guide.txt"

def main() -> None:
    if not os.getenv("OPENAI_API_KEY"):
        print("⚠️  OPENAI_API_KEY가 설정되지 않았습니다.")
        print("   .env 파일을 생성하고 API 키를 설정해주세요.")
        return
    
    print("🚀 RAG 파이프라인 테스트를 시작합니다...")
    
    rag = RAGPipeline(
        collection_name="test_collection",
        persist_directory="./test_chroma_db"
    )
    
    sample_file = create_sample_document()
    print(f"📄 샘플 문서 생성: {sample_file}")
    
    print("\n📚 문서를 벡터 데이터베이스에 추가 중...")
    result = rag.add_documents([sample_file])
    print(f"✅ 성공: {result['success']}")
    if result['failed']:
        print(f"❌ 실패: {result['failed']}")
    
    print(f"\n📊 총 {result['total_chunks']}개의 청크가 추가되었습니다.")
    
    stats = rag.get_stats()
    print(f"📈 데이터베이스 상태: {stats}")
    
    test_queries = [
        "머신러닝이 무엇인가요?",
        "딥러닝의 주요 모델들을 알려주세요",
        "RAG가 무엇인지 설명해주세요",
        "선형 회귀와 로지스틱 회귀의 차이점은?",
        "파이썬 프로그래밍에 대해 알려주세요"  # 문서에 없는 내용
    ]
    
    print("\n🤖 질문-답변 테스트:")
    print("=" * 60)
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n[질문 {i}] {query}")
        print("-" * 40)
        
        answer_data = rag.generate_answer(query, n_results=3)
        print(f"답변: {answer_data['answer']}")
        
        if answer_data['sources']:
            print(f"출처: {', '.join(answer_data['sources'])}")
        
        print(f"검색된 문서 수: {len(answer_data.get('search_results', []))}")

if __name__ == "__main__":
    main()