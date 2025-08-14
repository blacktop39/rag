import os
import logging
from typing import Dict, Any, List, Optional
from dotenv import load_dotenv
from langchain.prompts import ChatPromptTemplate

from .rag_pipeline import RAGPipeline
from .medical_validator import (
    MedicalContentValidator, 
    ConflictDetector, 
    create_safety_enhanced_prompt,
    RiskLevel,
    ReliabilityLevel
)

load_dotenv()
logger = logging.getLogger(__name__)

class MedicalChatbot:
    def __init__(self, 
                 collection_name: str = "medical_documents",
                 persist_directory: str = "./medical_chroma_db",
                 model_name: str = "gpt-3.5-turbo"):
        
        self.rag_pipeline = RAGPipeline(
            collection_name=collection_name,
            persist_directory=persist_directory,
            model_name=model_name
        )
        
        # 의료 검증 시스템 초기화
        self.content_validator = MedicalContentValidator()
        self.conflict_detector = ConflictDetector()
        
        self.medical_system_prompt = """당신은 전문적인 의료 정보 AI 어시스턴트입니다. 
제공된 의료 정보를 바탕으로 정확하고 도움이 되는 답변을 제공하세요.

⚠️ 중요한 안전 지침:
1. 개인별 진단이나 처방은 절대 하지 마세요
2. 심각한 증상의 경우 즉시 의료진 상담을 권하세요
3. 일반적인 의료 정보만 제공하고, 개인 맞춤 의료 조언은 피하세요
4. 응급상황 시 119 신고를 권하세요
5. 모든 답변 끝에 "정확한 진단과 치료를 위해서는 의료진과 상담하시기 바랍니다"를 포함하세요

컨텍스트 정보:
{context}

답변 시 다음을 포함하세요:
- 명확하고 이해하기 쉬운 설명
- 관련 증상이나 주의사항
- 일반적인 예방법이나 관리법
- 의료진 상담이 필요한 경우"""

    def load_medical_documents(self, file_paths: List[str]) -> Dict[str, Any]:
        """의료 문서들을 벡터 데이터베이스에 로드 (검증 포함)"""
        try:
            # 문서 로드 전 안전성 검증
            validation_results = []
            safe_files = []
            
            for file_path in file_paths:
                if os.path.exists(file_path):
                    # 파일 내용 읽기
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            content = f.read()
                        
                        # 내용 검증
                        metadata = {"source": file_path, "file_name": os.path.basename(file_path)}
                        validation = self.content_validator.validate_content(content, metadata)
                        validation_results.append({
                            "file": file_path,
                            "validation": validation
                        })
                        
                        if validation.is_safe:
                            safe_files.append(file_path)
                            logger.info(f"안전 검증 통과: {file_path}")
                        else:
                            logger.warning(f"안전 검증 실패: {file_path} - {validation.warnings}")
                    
                    except Exception as e:
                        logger.error(f"파일 검증 실패 {file_path}: {e}")
            
            # 안전한 파일들만 로드
            if safe_files:
                result = self.rag_pipeline.add_documents(safe_files)
                result["validation_results"] = validation_results
                result["safe_files"] = safe_files
                result["blocked_files"] = [f for f in file_paths if f not in safe_files]
                logger.info(f"의료 문서 로드 완료: {len(safe_files)}/{len(file_paths)} 파일")
            else:
                result = {
                    "success": [],
                    "failed": file_paths,
                    "total_chunks": 0,
                    "validation_results": validation_results,
                    "safe_files": [],
                    "blocked_files": file_paths
                }
                logger.warning("로드 가능한 안전한 파일이 없습니다")
            
            return result
            
        except Exception as e:
            logger.error(f"의료 문서 로드 실패: {e}")
            raise

    def ask_medical_question(self, question: str, n_results: int = 3) -> Dict[str, Any]:
        """의료 관련 질문에 대한 답변 생성 (향상된 안전 검증 포함)"""
        try:
            # 1. 응급상황 키워드 체크
            emergency_keywords = ['응급', '위급', '심각', '의식잃음', '호흡곤란', '가슴통증', '심장마비']
            if any(keyword in question for keyword in emergency_keywords):
                return {
                    "answer": "🚨 응급상황으로 보입니다. 즉시 119에 신고하거나 가장 가까운 응급실로 가시기 바랍니다. 이는 의료 응급상황일 수 있어 즉각적인 전문의료진의 도움이 필요합니다.",
                    "type": "emergency",
                    "sources": [],
                    "query": question,
                    "safety_level": "critical"
                }
            
            # 2. RAG 파이프라인을 통한 문서 검색
            search_results = self.rag_pipeline.search_documents(question, n_results)
            
            if not search_results:
                return {
                    "answer": "죄송합니다. 해당 의료 정보를 찾을 수 없습니다. 구체적인 의료 상담이 필요하시면 의료진과 상담하시기 바랍니다.",
                    "type": "no_result",
                    "sources": [],
                    "query": question,
                    "safety_level": "safe"
                }
            
            # 3. 검색된 콘텐츠 안전성 검증
            combined_content = "\n".join([result['content'] for result in search_results])
            content_validation = self.content_validator.validate_content(
                combined_content, 
                {"source": "rag_search_results"}
            )
            
            # 4. 충돌 감지
            conflict_info = self.conflict_detector.detect_conflicts(combined_content, question)
            
            # 5. 위험한 콘텐츠 감지 시 즉시 안전 응답
            if content_validation.risk_level == RiskLevel.DANGEROUS:
                warning_msg = "; ".join(content_validation.warnings)
                return {
                    "answer": f"""⚠️ 검색된 정보에 위험한 내용이 감지되었습니다.
                    
감지된 문제: {warning_msg}

안전을 위해 해당 정보는 제공하지 않습니다. 
정확한 의료 정보는 반드시 의료진과 직접 상담하시기 바랍니다.

🏥 권장사항: 병원 방문 또는 의료 상담 전화를 이용하세요.""",
                    "type": "blocked_dangerous",
                    "sources": [],
                    "query": question,
                    "safety_level": "blocked",
                    "validation_warnings": content_validation.warnings
                }
            
            # 6. 안전한 프롬프트 생성
            enhanced_prompt = create_safety_enhanced_prompt(content_validation, conflict_info)
            
            # 7. 컨텍스트 구성 (신뢰도 정보 포함)
            context_parts = []
            for result in search_results:
                source_info = result['metadata'].get('file_name', 'Unknown')
                reliability_info = ""
                
                # 신뢰도 정보 추가
                if content_validation.reliability == ReliabilityLevel.LOW:
                    reliability_info = " [신뢰도 낮음]"
                elif content_validation.reliability == ReliabilityLevel.HIGH:
                    reliability_info = " [신뢰할 만한 출처]"
                
                context_parts.append(f"[출처: {source_info}{reliability_info}]\n{result['content']}")
            
            context = "\n\n".join(context_parts)
            
            # 8. LLM 답변 생성
            prompt = ChatPromptTemplate.from_messages([
                ("system", enhanced_prompt),
                ("human", "질문: {question}")
            ])
            
            chain = prompt | self.rag_pipeline.llm
            
            response = chain.invoke({
                "context": context,
                "question": question
            })
            
            # 9. 추가 안전 메시지 구성
            safety_messages = ["⚠️ 정확한 진단과 치료를 위해서는 의료진과 상담하시기 바랍니다."]
            
            if content_validation.warnings:
                safety_messages.extend([f"• {warning}" for warning in content_validation.warnings])
            
            if conflict_info.get("has_conflicts"):
                safety_messages.append("• 일반적인 의학 지식과 다른 내용이 포함되어 있을 수 있습니다.")
            
            if content_validation.recommendations:
                safety_messages.extend([f"• {rec}" for rec in content_validation.recommendations])
            
            final_answer = response.content + "\n\n" + "\n".join(safety_messages)
            
            # 10. 응답 구성
            sources = list(set([
                result['metadata'].get('file_name', 'Unknown')
                for result in search_results
            ]))
            
            # 안전 레벨 결정
            safety_level = "safe"
            if content_validation.risk_level == RiskLevel.CAUTION:
                safety_level = "caution"
            elif content_validation.risk_level == RiskLevel.DANGEROUS:
                safety_level = "dangerous"
            
            return {
                "answer": final_answer,
                "type": "medical_info_validated",
                "sources": sources,
                "query": question,
                "search_results": search_results,
                "safety_level": safety_level,
                "validation_result": {
                    "risk_level": content_validation.risk_level.value,
                    "reliability": content_validation.reliability.value,
                    "confidence_score": content_validation.confidence_score,
                    "warnings": content_validation.warnings,
                    "recommendations": content_validation.recommendations
                },
                "conflict_info": conflict_info,
                "enhanced_safety": True
            }
            
        except Exception as e:
            logger.error(f"의료 질문 처리 중 오류: {e}")
            return {
                "answer": f"답변 생성 중 오류가 발생했습니다. 의료진과 직접 상담하시기 바랍니다. (오류: {str(e)})",
                "type": "error",
                "sources": [],
                "query": question,
                "safety_level": "error"
            }

    def get_available_topics(self) -> List[str]:
        """사용 가능한 의료 주제 목록 반환"""
        topics = [
            "고혈압 (Hypertension)",
            "당뇨병 (Diabetes Mellitus)", 
            "감기 (Common Cold)",
            "독감 (Influenza)",
            "고지혈증 (Hyperlipidemia)",
            "골다공증 (Osteoporosis)",
            "응급상황 대처법"
        ]
        return topics

    def get_stats(self) -> Dict[str, Any]:
        """의료 데이터베이스 통계 정보"""
        stats = self.rag_pipeline.get_stats()
        stats["available_topics"] = self.get_available_topics()
        return stats

    def interactive_chat(self):
        """대화형 의료 상담 챗봇"""
        print("🏥 의료 정보 AI 어시스턴트입니다.")
        print("=" * 50)
        print("⚠️  주의사항:")
        print("- 이는 일반적인 의료 정보 제공 서비스입니다")
        print("- 개인별 진단이나 처방은 제공하지 않습니다")
        print("- 심각한 증상이 있으시면 즉시 의료진과 상담하세요")
        print("- 응급상황 시 119에 신고하세요")
        print("=" * 50)
        
        available_topics = self.get_available_topics()
        print(f"\n📋 이용 가능한 의료 정보 주제:")
        for i, topic in enumerate(available_topics, 1):
            print(f"  {i}. {topic}")
        
        print(f"\n💬 질문을 입력하세요 (종료: 'quit' 또는 'exit'):")
        print("-" * 50)
        
        while True:
            try:
                user_input = input("\n👤 질문: ").strip()
                
                if user_input.lower() in ['quit', 'exit', '종료', '나가기']:
                    print("👋 의료 상담을 종료합니다. 건강하세요!")
                    break
                
                if not user_input:
                    continue
                
                print("🤖 답변 생성 중...")
                result = self.ask_medical_question(user_input)
                
                print(f"\n🩺 답변:")
                print("-" * 30)
                print(result["answer"])
                
                if result["sources"]:
                    print(f"\n📚 참고 자료: {', '.join(result['sources'])}")
                
                print(f"\n📊 답변 유형: {result['type']}")
                
            except KeyboardInterrupt:
                print("\n\n👋 의료 상담을 종료합니다. 건강하세요!")
                break
            except Exception as e:
                print(f"\n❌ 오류가 발생했습니다: {e}")
                print("다시 시도해주세요.")

def main():
    """메인 실행 함수"""
    if not os.getenv("OPENAI_API_KEY"):
        print("⚠️  OPENAI_API_KEY가 설정되지 않았습니다.")
        print("   .env 파일을 생성하고 API 키를 설정해주세요.")
        return
    
    # 의료 챗봇 초기화
    chatbot = MedicalChatbot()
    
    # 의료 문서 로드
    medical_file = "data/medical_info.txt"
    if os.path.exists(medical_file):
        print("📄 의료 문서를 로드하는 중...")
        result = chatbot.load_medical_documents([medical_file])
        print(f"✅ 로드 완료: {result['total_chunks']}개 문서 청크")
    else:
        print(f"❌ 의료 문서를 찾을 수 없습니다: {medical_file}")
        return
    
    # 대화형 챗봇 시작
    chatbot.interactive_chat()

if __name__ == "__main__":
    main()