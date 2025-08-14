# 🤖 RAG (Retrieval-Augmented Generation) 프로젝트

**OpenAI**와 **Ollama(로컬 LLM)** 모두 지원하는 한국어 최적화 RAG 시스템입니다.

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Poetry](https://img.shields.io/badge/Poetry-Package%20Manager-blue.svg)](https://python-poetry.org/)
[![Ollama](https://img.shields.io/badge/Ollama-Local%20LLM-green.svg)](https://ollama.ai/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## ✨ 주요 특징

### 🚀 **다중 LLM 지원**
- **OpenAI**: GPT-3.5, GPT-4 (클라우드)
- **Ollama**: Llama 3.2, Qwen2.5, Gemma2 등 (로컬)
- 실시간 모델 전환 가능

### 🔍 **강력한 검색 기능**
- ChromaDB 벡터 데이터베이스
- 다국어 임베딩 지원
- 유사도 기반 컨텍스트 검색

### 🏥 **의료 정보 특화**
- 의료 용어 최적화
- 안전 장치 내장 (응급상황 감지)
- 면책 조항 자동 추가

### 📱 **사용자 친화적**
- 대화형 챗봇 인터페이스
- 웹 API (FastAPI)
- Poetry 패키지 관리

## 🛠️ 설치 및 설정

### 1️⃣ 저장소 클론
```bash
git clone https://github.com/blacktop39/rag.git
cd rag
```

### 2️⃣ Poetry 설치 (미설치 시)
```bash
curl -sSL https://install.python-poetry.org | python3 -
```

### 3️⃣ 의존성 설치
```bash
poetry install
```

### 4️⃣ 환경 변수 설정
```bash
cp .env.example .env
```

`.env` 파일을 편집하여 사용할 LLM 설정:

**Option 1: OpenAI 사용**
```env
LLM_TYPE=openai
LLM_MODEL=gpt-3.5-turbo
OPENAI_API_KEY=your_api_key_here
```

**Option 2: Ollama (로컬) 사용**
```env
LLM_TYPE=ollama
LLM_MODEL=llama3.2
EMBEDDING_TYPE=sentence-transformer
```

## 🚀 사용법

### 🎯 빠른 시작

#### 1. 의료 챗봇 데모 (추천)
```bash
poetry run medical-demo
```

#### 2. 로컬 LLM 테스트
```bash
# Ollama 설치 후 모델 다운로드
ollama pull llama3.2
ollama serve

# 로컬 LLM 데모 실행
poetry run local-llm-demo
```

#### 3. 간단한 Ollama 테스트
```bash
poetry run python simple_ollama_test.py
```

### 📚 프로그래밍 사용 예제

#### OpenAI 사용
```python
from src.rag_pipeline import RAGPipeline

rag = RAGPipeline(
    llm_type="openai",
    llm_model="gpt-3.5-turbo"
)

result = rag.add_documents(["data/medical_info.txt"])
answer = rag.generate_answer("고혈압이 무엇인가요?")
print(answer["answer"])
```

#### 로컬 LLM 사용
```python
from src.enhanced_rag_pipeline import EnhancedRAGPipeline

rag = EnhancedRAGPipeline(
    llm_type="ollama",
    llm_model="llama3.2",
    embedding_type="sentence-transformer"
)

result = rag.add_documents(["data/medical_info.txt"])
answer = rag.generate_answer("당뇨병 증상은?")
```

## 프로젝트 구조

```
rag/
├── src/
│   ├── __init__.py
│   ├── vector_store.py      # ChromaDB 벡터 저장소
│   ├── document_processor.py # 문서 처리 및 임베딩
│   └── rag_pipeline.py      # RAG 파이프라인
├── tests/
│   └── test_rag.py          # 단위 테스트
├── data/                    # 문서 파일 저장소
├── pyproject.toml           # Poetry 의존성 및 설정
├── .env.example            # 환경 변수 예제
└── example_usage.py        # 사용 예제
```

## 주요 구성 요소

### VectorStore
- ChromaDB를 사용한 벡터 데이터베이스
- 문서 임베딩 저장 및 유사도 검색

### DocumentProcessor  
- 문서 로드 (PDF, TXT)
- 텍스트 청킹 및 분할
- OpenAI 임베딩 생성

### RAGPipeline
- 전체 RAG 워크플로우 관리
- 문서 추가, 검색, 답변 생성

## 테스트

```bash
poetry run pytest tests/
```

## 개발 도구

코드 포맷팅:
```bash
poetry run black .
poetry run isort .
```

타입 체크:
```bash
poetry run mypy src/
```

린트:
```bash
poetry run flake8 src/
```

## 🛠️ 스크립트 명령어

| 명령어 | 설명 |
|--------|------|
| `poetry run medical-demo` | 의료 정보 챗봇 데모 (추천) |
| `poetry run local-llm-demo` | 로컬 LLM 대화형 데모 |
| `poetry run medical-chatbot` | 의료 챗봇 직접 실행 |
| `poetry run rag-example` | 기본 RAG 예제 |

## 🦙 Ollama 설치 가이드

자세한 Ollama 설치 및 설정은 [OLLAMA_SETUP.md](OLLAMA_SETUP.md)를 참조하세요.

### 추천 모델
- **llama3.2** (3B): 균형잡힌 성능, 8GB RAM
- **qwen2.5:3b** (3B): 한국어 우수, 6GB RAM  
- **gemma2:2b** (2B): 초경량, 4GB RAM

```bash
# 모델 설치 예제
ollama pull llama3.2
ollama pull qwen2.5:3b
ollama pull gemma2:2b
```

## ⚙️ 환경 변수

| 변수명 | 설명 | 기본값 |
|--------|------|--------|
| `LLM_TYPE` | LLM 타입 (openai/ollama) | openai |
| `LLM_MODEL` | 사용할 모델명 | gpt-3.5-turbo |
| `EMBEDDING_TYPE` | 임베딩 타입 | openai |
| `EMBEDDING_MODEL` | 임베딩 모델명 | text-embedding-3-small |
| `OPENAI_API_KEY` | OpenAI API 키 | 필수 (OpenAI 사용시) |
| `OLLAMA_BASE_URL` | Ollama 서버 URL | http://localhost:11434 |

## 🔧 개발 도구

```bash
# 코드 포맷팅
poetry run black .
poetry run isort .

# 타입 체크
poetry run mypy src/

# 테스트 실행
poetry run pytest

# 린트
poetry run flake8 src/
```

## 🤝 기여하기

1. Fork 이 저장소
2. Feature 브랜치 생성 (`git checkout -b feature/amazing-feature`)
3. 변경사항 커밋 (`git commit -m 'Add amazing feature'`)
4. 브랜치에 Push (`git push origin feature/amazing-feature`)
5. Pull Request 생성

## 📄 라이선스

이 프로젝트는 MIT 라이선스 하에 배포됩니다. 자세한 내용은 [LICENSE](LICENSE) 파일을 참조하세요.

## 🙋‍♂️ 지원

- 이슈: [GitHub Issues](https://github.com/blacktop39/rag/issues)
- 토론: [GitHub Discussions](https://github.com/blacktop39/rag/discussions)

## ⭐ 별표 부탁드려요!

이 프로젝트가 도움이 되셨다면 ⭐ 별표를 눌러주세요!