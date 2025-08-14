#!/usr/bin/env python3
"""
간단한 Ollama 테스트 (SSL 문제 회피용)
"""

import os
import sys
import requests
import ssl
import urllib3

# SSL 경고 무시 (개발 환경용)
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

def test_ollama_connection():
    """Ollama 연결 테스트"""
    try:
        print("🔍 Ollama 서버 연결 테스트...")
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        
        if response.status_code == 200:
            data = response.json()
            models = data.get("models", [])
            
            print("✅ Ollama 서버 연결 성공!")
            print(f"📋 설치된 모델 ({len(models)}개):")
            
            for model in models:
                name = model.get("name", "Unknown")
                size_gb = round(model.get("size", 0) / (1024**3), 1)
                family = model.get("details", {}).get("family", "Unknown")
                print(f"   - {name} ({family}, {size_gb}GB)")
            
            return models
        else:
            print(f"❌ Ollama 서버 응답 오류: {response.status_code}")
            return []
            
    except Exception as e:
        print(f"❌ Ollama 연결 실패: {e}")
        return []

def test_ollama_chat(model_name="gemma2:2b"):
    """Ollama 채팅 테스트"""
    try:
        print(f"\n🤖 {model_name} 모델 테스트...")
        
        payload = {
            "model": model_name,
            "messages": [
                {
                    "role": "system", 
                    "content": "당신은 도움이 되는 한국어 AI 어시스턴트입니다."
                },
                {
                    "role": "user", 
                    "content": "안녕하세요!"
                }
            ],
            "stream": False
        }
        
        print("💭 응답 생성 중...")
        response = requests.post(
            "http://localhost:11434/api/chat",
            json=payload,
            timeout=120  # 2분으로 연장
        )
        
        if response.status_code == 200:
            data = response.json()
            answer = data.get("message", {}).get("content", "응답 없음")
            
            print("✅ 응답 생성 성공!")
            print(f"🩺 답변: {answer}")
            return True
        else:
            print(f"❌ 채팅 요청 실패: {response.status_code}")
            print(f"응답: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ 채팅 테스트 실패: {e}")
        return False

def simple_rag_test():
    """간단한 RAG 테스트 (외부 라이브러리 최소화)"""
    print("\n📚 간단한 RAG 테스트")
    print("=" * 40)
    
    # 간단한 의료 지식 베이스
    knowledge_base = {
        "고혈압": "고혈압은 혈압이 정상 범위보다 지속적으로 높은 상태입니다. 수축기 혈압 140mmHg 이상, 이완기 혈압 90mmHg 이상일 때 진단됩니다. 주요 원인으로는 유전, 나이, 비만, 짠 음식 섭취, 운동 부족 등이 있습니다.",
        "당뇨병": "당뇨병은 혈당 조절에 문제가 생기는 대사 질환입니다. 제1형은 인슐린을 생산하지 못하고, 제2형은 인슐린 저항성이 주된 원인입니다. 주요 증상으로는 다뇨, 다음, 다식, 체중 감소 등이 있습니다.",
        "감기": "감기는 바이러스에 의한 상부 호흡기 감염입니다. 주요 증상으로는 콧물, 코막힘, 재채기, 기침, 목 아픔 등이 있으며, 보통 7-10일 내에 자연 치유됩니다."
    }
    
    def simple_search(query, knowledge_base):
        """간단한 키워드 검색"""
        results = []
        for topic, content in knowledge_base.items():
            if topic in query or any(keyword in content for keyword in query.split()):
                results.append((topic, content))
        return results
    
    # 테스트 질문들
    test_questions = [
        "고혈압이 무엇인가요?",
        "당뇨병 증상을 알려주세요",
        "감기는 어떻게 치료하나요?"
    ]
    
    for question in test_questions:
        print(f"\n❓ 질문: {question}")
        
        # 1. 간단한 검색
        search_results = simple_search(question, knowledge_base)
        
        if search_results:
            context = "\n".join([f"[{topic}] {content}" for topic, content in search_results])
            print(f"📖 찾은 정보: {len(search_results)}개")
            
            # 2. Ollama로 답변 생성
            payload = {
                "model": "gemma2:2b",
                "messages": [
                    {
                        "role": "system",
                        "content": f"""당신은 의료 정보 AI 어시스턴트입니다. 다음 의료 정보를 바탕으로 질문에 답변하세요.

의료 정보:
{context}

주의사항: 일반적인 의료 정보만 제공하고, 개인별 진단은 하지 마세요."""
                    },
                    {
                        "role": "user",
                        "content": question
                    }
                ],
                "stream": False
            }
            
            try:
                response = requests.post(
                    "http://localhost:11434/api/chat",
                    json=payload,
                    timeout=30
                )
                
                if response.status_code == 200:
                    data = response.json()
                    answer = data.get("message", {}).get("content", "응답 없음")
                    print(f"🤖 답변: {answer[:200]}...")
                else:
                    print(f"❌ 응답 생성 실패: {response.status_code}")
                    
            except Exception as e:
                print(f"❌ 오류: {e}")
        else:
            print("❌ 관련 정보를 찾을 수 없습니다.")

def main():
    """메인 함수"""
    print("🦙 간단한 Ollama + RAG 테스트")
    print("=" * 50)
    
    # 1. Ollama 연결 테스트
    models = test_ollama_connection()
    
    if not models:
        print("\n❌ Ollama가 실행되지 않았거나 모델이 없습니다.")
        print("다음 명령어를 실행해보세요:")
        print("1. ollama serve  # 다른 터미널에서")
        print("2. ollama pull gemma2:2b")
        return
    
    # 2. 채팅 테스트
    model_name = models[0]["name"]
    chat_success = test_ollama_chat(model_name)
    
    if chat_success:
        # 3. 간단한 RAG 테스트
        simple_rag_test()
    
    print(f"\n✅ 테스트 완료!")
    print("더 고급 기능을 위해서는 SSL 인증서 문제를 해결하거나")
    print("OpenAI API를 사용하는 것을 권장합니다.")

if __name__ == "__main__":
    main()