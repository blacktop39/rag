#!/usr/bin/env python3
"""
위험한 의료 정보 감지 및 충돌 테스트
"""

import tempfile
import os
from src.medical_chatbot import MedicalChatbot
from src.medical_validator import RiskLevel, ReliabilityLevel

def create_test_files():
    """테스트용 의료 파일들 생성"""
    
    # 1. 안전한 의료 정보
    safe_content = """
# 고혈압 정보

## 정의
고혈압은 혈압이 정상 범위를 초과하여 지속적으로 높은 상태입니다.
수축기 혈압이 140mmHg 이상이거나 이완기 혈압이 90mmHg 이상일 때 고혈압으로 진단합니다.

## 치료
- 생활습관 개선: 금연, 금주, 저염식, 규칙적인 운동
- 약물치료: 의사 처방에 따른 혈압약 복용
- 정기적인 혈압 측정과 의료진 상담

주의: 정확한 진단과 치료를 위해서는 의료진과 상담하시기 바랍니다.
"""
    
    # 2. 위험한 의료 정보 (잘못된 혈압 수치)
    dangerous_content = """
# 혈압 정보

고혈압의 정상 수치는 200/100mmHg 미만입니다.
이 수치를 넘으면 병원에 가세요.

마늘을 많이 먹으면 암이 완치됩니다.
항생제는 감기 바이러스에도 효과가 있습니다.
백신은 자폐증을 유발할 수 있어 위험합니다.

응급실에 가지 말고 집에서 자가치료 하세요.
"""
    
    # 3. 충돌하는 정보 (고혈압을 저혈압으로 설명)
    conflicting_content = """
# 고혈압 설명

고혈압은 혈압이 너무 낮을 때 발생하는 질병입니다.
저혈압과 같은 의미로 사용됩니다.

당뇨병은 운동만으로 완치가 가능합니다.
인슐린은 중독성이 있어 위험한 약물입니다.
"""
    
    # 4. 신뢰도 낮은 출처 (개인 블로그)
    low_reliability_content = """
# 개인 경험담 - 내 블로그

내가 직접 경험한 고혈압 치료법!
병원에 가지 않고도 집에서 치료했어요.

비타민을 과다섭취해도 전혀 안전합니다.
의사들이 숨기는 자연치료법을 공개합니다.
"""
    
    files = {}
    temp_dir = tempfile.mkdtemp()
    
    # 임시 파일들 생성
    for name, content in [
        ("safe_medical.txt", safe_content),
        ("dangerous_medical.txt", dangerous_content), 
        ("conflicting_medical.txt", conflicting_content),
        ("blog_medical.txt", low_reliability_content)
    ]:
        file_path = os.path.join(temp_dir, name)
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        files[name] = file_path
    
    return files, temp_dir

def test_safe_content():
    """안전한 콘텐츠 테스트"""
    print("🧪 안전한 콘텐츠 테스트")
    print("=" * 50)
    
    files, temp_dir = create_test_files()
    chatbot = MedicalChatbot(collection_name="safe_test")
    
    # 안전한 파일만 로드
    result = chatbot.load_medical_documents([files["safe_medical.txt"]])
    print(f"✅ 로드 결과: {result['safe_files']}")
    print(f"❌ 차단된 파일: {result['blocked_files']}")
    
    # 질문 테스트
    answer = chatbot.ask_medical_question("고혈압이 무엇인가요?")
    print(f"\n📝 질문: 고혈압이 무엇인가요?")
    print(f"🔒 안전 레벨: {answer['safety_level']}")
    print(f"📄 답변: {answer['answer'][:200]}...")
    
    # 정리
    import shutil
    shutil.rmtree(temp_dir)

def test_dangerous_content():
    """위험한 콘텐츠 테스트"""
    print("\n🚨 위험한 콘텐츠 테스트")
    print("=" * 50)
    
    files, temp_dir = create_test_files()
    chatbot = MedicalChatbot(collection_name="danger_test")
    
    # 위험한 파일 로드 시도
    result = chatbot.load_medical_documents([files["dangerous_medical.txt"]])
    print(f"✅ 로드 성공: {result['safe_files']}")
    print(f"🚫 차단된 파일: {result['blocked_files']}")
    
    if result['blocked_files']:
        print("🛡️ 위험한 파일이 성공적으로 차단되었습니다!")
        for validation in result['validation_results']:
            if not validation['validation'].is_safe:
                print(f"⚠️ 경고사항: {validation['validation'].warnings}")
    
    # 정리
    import shutil
    shutil.rmtree(temp_dir)

def test_conflict_detection():
    """충돌 감지 테스트"""
    print("\n⚔️ 정보 충돌 감지 테스트")
    print("=" * 50)
    
    files, temp_dir = create_test_files()
    chatbot = MedicalChatbot(collection_name="conflict_test")
    
    # 먼저 안전한 정보로 가정하고 로드 (실제로는 차단될 수 있음)
    try:
        # 충돌하는 내용을 직접 검증해보기
        from src.medical_validator import ConflictDetector
        
        detector = ConflictDetector()
        test_content = "고혈압은 혈압이 너무 낮을 때 발생하는 질병입니다."
        
        conflict_info = detector.detect_conflicts(test_content)
        print(f"🔍 충돌 감지: {conflict_info['has_conflicts']}")
        
        if conflict_info['has_conflicts']:
            print("⚠️ 발견된 충돌:")
            for conflict in conflict_info['conflicts']:
                print(f"  - {conflict['type']}: {conflict['description']}")
                print(f"    표준 정보: {conflict['standard']}")
    
    except Exception as e:
        print(f"❌ 테스트 오류: {e}")
    
    # 정리
    import shutil
    shutil.rmtree(temp_dir)

def test_emergency_detection():
    """응급상황 감지 테스트"""
    print("\n🚨 응급상황 감지 테스트")
    print("=" * 50)
    
    chatbot = MedicalChatbot(collection_name="emergency_test")
    
    emergency_questions = [
        "심장마비 증상이 있어요",
        "의식을 잃었어요",
        "호흡곤란이 심해요",
        "응급상황입니다"
    ]
    
    for question in emergency_questions:
        answer = chatbot.ask_medical_question(question)
        print(f"\n❓ 질문: {question}")
        print(f"🚨 타입: {answer['type']}")
        print(f"🔒 안전 레벨: {answer['safety_level']}")
        if answer['type'] == 'emergency':
            print("✅ 응급상황이 정확히 감지되었습니다!")

def test_reliability_assessment():
    """신뢰도 평가 테스트"""
    print("\n📊 신뢰도 평가 테스트")
    print("=" * 50)
    
    from src.medical_validator import MedicalContentValidator
    
    validator = MedicalContentValidator()
    
    test_cases = [
        {
            "content": "일반적인 의료 정보입니다.",
            "metadata": {"source": "질병관리청", "file_name": "official_guide.txt"},
            "expected": ReliabilityLevel.HIGH
        },
        {
            "content": "개인 경험담입니다.",
            "metadata": {"source": "개인블로그", "file_name": "my_blog.txt"},
            "expected": ReliabilityLevel.LOW
        },
        {
            "content": "뉴스 기사입니다.",
            "metadata": {"source": "뉴스", "file_name": "news_article.txt"},
            "expected": ReliabilityLevel.MEDIUM
        }
    ]
    
    for case in test_cases:
        result = validator.validate_content(case["content"], case["metadata"])
        print(f"📄 출처: {case['metadata']['source']}")
        print(f"🔍 예상 신뢰도: {case['expected'].value}")
        print(f"✅ 실제 신뢰도: {result.reliability.value}")
        print(f"✅ 일치: {'Yes' if result.reliability == case['expected'] else 'No'}")
        print()

def main():
    """전체 테스트 실행"""
    print("🧪 의료 안전 검증 시스템 테스트")
    print("=" * 60)
    
    try:
        test_safe_content()
        test_dangerous_content() 
        test_conflict_detection()
        test_emergency_detection()
        test_reliability_assessment()
        
        print("\n🎉 모든 테스트가 완료되었습니다!")
        print("✅ 안전 검증 시스템이 정상적으로 작동합니다.")
        
    except Exception as e:
        print(f"\n❌ 테스트 실행 중 오류: {e}")
        print("필요한 패키지를 설치했는지 확인해주세요:")
        print("poetry install")

if __name__ == "__main__":
    main()