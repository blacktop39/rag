#!/usr/bin/env python3
"""
의료 정보 RAG 챗봇 데모
"""

import os
import logging
from dotenv import load_dotenv
from src.medical_chatbot import MedicalChatbot

logging.basicConfig(level=logging.INFO)
load_dotenv()

def demo_medical_questions():
    """의료 질문 데모"""
    demo_questions = [
        "고혈압이 무엇인가요?",
        "당뇨병의 증상은 어떤 것들이 있나요?", 
        "감기와 독감의 차이점을 알려주세요",
        "골다공증 예방방법이 있나요?",
        "혈압이 150/95인데 괜찮나요?",
        "응급상황에서는 어떻게 해야 하나요?",
        "심장마비 증상이 있어요"  # 응급상황 테스트
    ]
    
    return demo_questions

def run_demo():
    """데모 실행"""
    if not os.getenv("OPENAI_API_KEY"):
        print("⚠️  OPENAI_API_KEY가 설정되지 않았습니다.")
        print("   .env 파일을 생성하고 API 키를 설정해주세요.")
        return
    
    print("🏥 의료 정보 RAG 챗봇 데모를 시작합니다!")
    print("=" * 60)
    
    # 의료 챗봇 초기화
    chatbot = MedicalChatbot()
    
    # 의료 문서 로드
    medical_file = "data/medical_info.txt"
    if not os.path.exists(medical_file):
        print(f"❌ 의료 문서를 찾을 수 없습니다: {medical_file}")
        return
    
    print("📄 의료 문서를 벡터 데이터베이스에 로드 중...")
    result = chatbot.load_medical_documents([medical_file])
    
    if result['success']:
        print(f"✅ 문서 로드 성공: {result['total_chunks']}개 청크")
    else:
        print(f"❌ 문서 로드 실패: {result['failed']}")
        return
    
    # 데이터베이스 상태 확인
    stats = chatbot.get_stats()
    print(f"📊 데이터베이스 정보:")
    print(f"   - 컬렉션: {stats['collection_name']}")
    print(f"   - 총 문서 수: {stats['total_documents']}")
    print(f"   - 사용 가능한 주제: {len(stats['available_topics'])}개")
    
    # 데모 질문들 실행
    demo_questions = demo_medical_questions()
    
    print(f"\n🤖 {len(demo_questions)}개의 데모 질문을 테스트합니다:")
    print("=" * 60)
    
    for i, question in enumerate(demo_questions, 1):
        print(f"\n[질문 {i}] {question}")
        print("-" * 40)
        
        try:
            answer_data = chatbot.ask_medical_question(question)
            
            print(f"답변 유형: {answer_data['type']}")
            print(f"답변: {answer_data['answer']}")
            
            if answer_data.get('sources'):
                print(f"출처: {', '.join(answer_data['sources'])}")
            
            if answer_data.get('search_results'):
                print(f"검색된 문서 수: {len(answer_data['search_results'])}")
                
        except Exception as e:
            print(f"❌ 오류 발생: {e}")
    
    # 대화형 모드 안내
    print(f"\n" + "=" * 60)
    print("🗣️  대화형 모드를 시작하시겠습니까?")
    print("   'y' 또는 'yes'를 입력하면 대화형 챗봇이 시작됩니다.")
    print("   다른 키를 누르면 데모를 종료합니다.")
    
    user_choice = input("\n선택: ").strip().lower()
    
    if user_choice in ['y', 'yes', 'ㅇ', '네', '예']:
        print("\n🎯 대화형 의료 상담 모드를 시작합니다...")
        chatbot.interactive_chat()
    else:
        print("\n👋 의료 챗봇 데모를 종료합니다. 건강하세요!")

if __name__ == "__main__":
    run_demo()