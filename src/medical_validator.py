import re
import logging
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

class ReliabilityLevel(Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    UNKNOWN = "unknown"

class RiskLevel(Enum):
    SAFE = "safe"
    CAUTION = "caution"
    DANGEROUS = "dangerous"

@dataclass
class ValidationResult:
    is_safe: bool
    risk_level: RiskLevel
    reliability: ReliabilityLevel
    warnings: List[str]
    recommendations: List[str]
    confidence_score: float

class MedicalContentValidator:
    """의료 콘텐츠 검증 및 안전성 평가 시스템"""
    
    def __init__(self):
        self.dangerous_patterns = self._load_dangerous_patterns()
        self.medical_misinformation = self._load_misinformation_patterns()
        self.emergency_keywords = self._load_emergency_keywords()
        self.trusted_sources = self._load_trusted_sources()
        
    def _load_dangerous_patterns(self) -> List[Dict[str, Any]]:
        """위험한 의료 정보 패턴들"""
        return [
            {
                "pattern": r"(?:암|cancer).*(?:완치|치료).*(?:마늘|생강|민간요법)",
                "risk": RiskLevel.DANGEROUS,
                "message": "검증되지 않은 암 치료법 정보"
            },
            {
                "pattern": r"(?:혈압|blood pressure).*200.*정상",
                "risk": RiskLevel.DANGEROUS,
                "message": "잘못된 혈압 정상 수치 정보"
            },
            {
                "pattern": r"(?:당뇨병|diabetes).*완치.*(?:운동|식이)만으로",
                "risk": RiskLevel.DANGEROUS,
                "message": "당뇨병 완치 관련 잘못된 정보"
            },
            {
                "pattern": r"(?:항생제|antibiotic).*바이러스.*효과",
                "risk": RiskLevel.DANGEROUS,
                "message": "항생제와 바이러스에 대한 잘못된 정보"
            },
            {
                "pattern": r"(?:백신|vaccine).*(?:자폐|autism|위험)",
                "risk": RiskLevel.DANGEROUS,
                "message": "백신에 대한 잘못된 정보"
            },
            {
                "pattern": r"(?:병원|의사).*(?:필요없|불필요).*자가치료",
                "risk": RiskLevel.DANGEROUS,
                "message": "의료진 상담을 배제하는 위험한 조언"
            },
            {
                "pattern": r"(?:응급실|119).*(?:가지마|피해)",
                "risk": RiskLevel.DANGEROUS,
                "message": "응급상황에서 위험한 조언"
            }
        ]
    
    def _load_misinformation_patterns(self) -> List[Dict[str, Any]]:
        """의료 허위정보 패턴들"""
        return [
            {
                "pattern": r"(?:고혈압|hypertension).*(?:낮은|저혈압)",
                "category": "정의 오류",
                "correction": "고혈압은 혈압이 높은 상태입니다"
            },
            {
                "pattern": r"(?:감기|cold).*항생제.*필요",
                "category": "치료법 오류", 
                "correction": "감기는 바이러스 감염으로 항생제가 효과없습니다"
            },
            {
                "pattern": r"(?:인슐린|insulin).*중독.*위험",
                "category": "약물 오해",
                "correction": "인슐린은 당뇨병 치료에 필수적인 약물입니다"
            },
            {
                "pattern": r"(?:비타민|vitamin).*과다섭취.*안전",
                "category": "영양소 오해",
                "correction": "일부 비타민은 과다섭취 시 부작용이 있습니다"
            }
        ]
    
    def _load_emergency_keywords(self) -> List[str]:
        """응급상황 키워드들"""
        return [
            "심장마비", "심정지", "의식잃음", "호흡곤란", "심한가슴통증",
            "대량출혈", "골절", "화상", "중독", "알레르기쇼크",
            "뇌졸중", "경련", "발작", "응급", "위급", "심각"
        ]
    
    def _load_trusted_sources(self) -> List[str]:
        """신뢰할 수 있는 출처들"""
        return [
            "질병관리청", "대한의학회", "대한의사협회", "보건복지부",
            "WHO", "CDC", "Mayo Clinic", "WebMD", "의학논문",
            "병원공식자료", "의과대학", "간호대학"
        ]
    
    def validate_content(self, content: str, metadata: Optional[Dict] = None) -> ValidationResult:
        """의료 콘텐츠 종합 검증"""
        warnings = []
        recommendations = []
        risk_level = RiskLevel.SAFE
        reliability = ReliabilityLevel.UNKNOWN
        confidence_score = 0.8
        
        # 1. 위험한 패턴 검사
        dangerous_found = self._check_dangerous_patterns(content)
        if dangerous_found:
            risk_level = RiskLevel.DANGEROUS
            warnings.extend([d["message"] for d in dangerous_found])
            recommendations.append("전문의와 반드시 상담하세요")
            confidence_score -= 0.4
        
        # 2. 허위정보 패턴 검사
        misinformation_found = self._check_misinformation(content)
        if misinformation_found:
            if risk_level != RiskLevel.DANGEROUS:
                risk_level = RiskLevel.CAUTION
            warnings.extend([f"잠재적 오류: {m['category']}" for m in misinformation_found])
            recommendations.extend([m["correction"] for m in misinformation_found])
            confidence_score -= 0.2
        
        # 3. 출처 신뢰도 평가
        if metadata:
            reliability = self._assess_source_reliability(metadata)
            if reliability == ReliabilityLevel.LOW:
                if risk_level == RiskLevel.SAFE:
                    risk_level = RiskLevel.CAUTION
                warnings.append("신뢰도가 낮은 출처입니다")
                confidence_score -= 0.3
        
        # 4. 응급상황 키워드 확인
        emergency_found = self._check_emergency_keywords(content)
        if emergency_found:
            warnings.append("응급상황 관련 내용이 감지되었습니다")
            recommendations.append("즉시 119에 신고하거나 응급실을 방문하세요")
        
        # 5. 종합 판단
        is_safe = risk_level != RiskLevel.DANGEROUS
        
        return ValidationResult(
            is_safe=is_safe,
            risk_level=risk_level,
            reliability=reliability,
            warnings=warnings,
            recommendations=recommendations,
            confidence_score=max(0.1, confidence_score)
        )
    
    def _check_dangerous_patterns(self, content: str) -> List[Dict[str, Any]]:
        """위험한 패턴 검사"""
        found_patterns = []
        content_lower = content.lower()
        
        for pattern_info in self.dangerous_patterns:
            if re.search(pattern_info["pattern"], content_lower, re.IGNORECASE):
                found_patterns.append(pattern_info)
                logger.warning(f"위험한 패턴 감지: {pattern_info['message']}")
        
        return found_patterns
    
    def _check_misinformation(self, content: str) -> List[Dict[str, Any]]:
        """허위정보 패턴 검사"""
        found_misinformation = []
        content_lower = content.lower()
        
        for misinfo in self.medical_misinformation:
            if re.search(misinfo["pattern"], content_lower, re.IGNORECASE):
                found_misinformation.append(misinfo)
                logger.warning(f"허위정보 패턴 감지: {misinfo['category']}")
        
        return found_misinformation
    
    def _check_emergency_keywords(self, content: str) -> List[str]:
        """응급상황 키워드 검사"""
        found_keywords = []
        content_lower = content.lower()
        
        for keyword in self.emergency_keywords:
            if keyword in content_lower:
                found_keywords.append(keyword)
        
        return found_keywords
    
    def _assess_source_reliability(self, metadata: Dict) -> ReliabilityLevel:
        """출처 신뢰도 평가"""
        source = metadata.get("source", "").lower()
        file_name = metadata.get("file_name", "").lower()
        
        # 신뢰할 수 있는 출처 확인
        for trusted in self.trusted_sources:
            if trusted.lower() in source or trusted.lower() in file_name:
                return ReliabilityLevel.HIGH
        
        # 블로그, 카페 등 개인 출처
        personal_sources = ["blog", "cafe", "개인", "후기", "경험담"]
        for personal in personal_sources:
            if personal in source or personal in file_name:
                return ReliabilityLevel.LOW
        
        # 뉴스, 잡지 등 중간 수준
        media_sources = ["news", "뉴스", "잡지", "magazine"]
        for media in media_sources:
            if media in source or media in file_name:
                return ReliabilityLevel.MEDIUM
        
        return ReliabilityLevel.UNKNOWN

class ConflictDetector:
    """RAG 검색 결과와 일반 의학 지식 간 충돌 감지"""
    
    def __init__(self):
        self.known_medical_facts = self._load_medical_facts()
    
    def _load_medical_facts(self) -> Dict[str, str]:
        """알려진 의학 사실들"""
        return {
            "고혈압_정의": "혈압이 정상보다 높은 상태 (140/90mmHg 이상)",
            "당뇨병_완치": "당뇨병은 완치되지 않고 관리하는 질병",
            "감기_항생제": "감기는 바이러스 감염으로 항생제 효과 없음",
            "정상체온": "36.5-37.5도 사이가 정상 체온",
            "응급실_방문": "의식잃음, 호흡곤란, 심한 흉통 시 즉시 응급실"
        }
    
    def detect_conflicts(self, rag_content: str, topic: str = "") -> Dict[str, Any]:
        """내용 충돌 감지"""
        conflicts = []
        suggestions = []
        
        content_lower = rag_content.lower()
        
        # 각 의학 사실과 대조
        for fact_key, fact_value in self.known_medical_facts.items():
            conflict_info = self._check_specific_conflict(content_lower, fact_key, fact_value)
            if conflict_info:
                conflicts.append(conflict_info)
                suggestions.append(f"일반적으로 알려진 정보: {fact_value}")
        
        return {
            "has_conflicts": len(conflicts) > 0,
            "conflicts": conflicts,
            "suggestions": suggestions,
            "confidence": 0.9 if conflicts else 0.1
        }
    
    def _check_specific_conflict(self, content: str, fact_key: str, fact_value: str) -> Optional[Dict]:
        """특정 사실과의 충돌 확인"""
        if fact_key == "고혈압_정의":
            if "고혈압" in content and ("낮은" in content or "저혈압" in content):
                return {
                    "type": "정의 충돌",
                    "description": "고혈압을 낮은 혈압으로 설명",
                    "standard": fact_value
                }
        
        elif fact_key == "당뇨병_완치":
            if "당뇨병" in content and "완치" in content and ("가능" in content or "된다" in content):
                return {
                    "type": "치료 정보 충돌",
                    "description": "당뇨병 완치 가능하다고 주장",
                    "standard": fact_value
                }
        
        elif fact_key == "감기_항생제":
            if "감기" in content and "항생제" in content and ("효과" in content or "치료" in content):
                return {
                    "type": "치료법 충돌",
                    "description": "감기에 항생제가 효과적이라고 주장",
                    "standard": fact_value
                }
        
        return None

def create_safety_enhanced_prompt(validation_result: ValidationResult, 
                                 conflict_info: Dict[str, Any] = None) -> str:
    """검증 결과를 반영한 안전 강화 프롬프트 생성"""
    
    base_prompt = """당신은 의료 정보 AI 어시스턴트입니다. 다음 중요 사항을 반드시 준수하세요:

⚠️ 안전 지침:
1. 개인별 진단이나 처방은 절대 하지 마세요
2. 심각한 증상의 경우 즉시 의료진 상담을 권하세요
3. 응급상황 시 119 신고를 최우선으로 안내하세요"""

    # 검증 결과에 따른 추가 지침
    if validation_result.risk_level == RiskLevel.DANGEROUS:
        base_prompt += """

🚨 위험 경고:
- 제공된 문서에 위험한 정보가 포함되어 있습니다
- 해당 정보는 신뢰하지 마세요
- 반드시 전문의와 상담하도록 안내하세요"""
    
    elif validation_result.risk_level == RiskLevel.CAUTION:
        base_prompt += """

⚠️ 주의 사항:
- 제공된 정보의 신뢰도가 확실하지 않습니다
- 일반적인 의학 상식과 함께 제공하세요
- 전문의 확인을 권하세요"""
    
    # 충돌 감지 시 추가 지침
    if conflict_info and conflict_info.get("has_conflicts"):
        base_prompt += """

🔍 정보 충돌 감지:
- 문서 정보와 일반 의학 지식이 상충됩니다
- 양쪽 정보를 모두 언급하고 차이점을 설명하세요
- 정확한 정보는 의료진에게 확인받도록 안내하세요"""

    if validation_result.warnings:
        warnings_text = "\n".join([f"- {warning}" for warning in validation_result.warnings])
        base_prompt += f"""

⚠️ 감지된 경고사항:
{warnings_text}"""

    base_prompt += """

컨텍스트 정보:
{context}

답변 시 반드시 포함할 내용:
- 명확하고 이해하기 쉬운 설명
- 관련 주의사항
- "정확한 진단과 치료를 위해서는 의료진과 상담하시기 바랍니다" 문구"""

    return base_prompt