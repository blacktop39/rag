# Ollama 설치 및 사용 가이드

## Ollama란?
Ollama는 Llama 3.2, Qwen, Gemma 등의 대형 언어 모델을 로컬에서 실행할 수 있게 해주는 도구입니다.

## 1. Ollama 설치

### Windows
1. [Ollama 공식 사이트](https://ollama.ai/download)에서 Windows 버전 다운로드
2. 설치 파일 실행 후 설치 완료
3. 터미널에서 `ollama --version`으로 설치 확인

### macOS
```bash
# Homebrew 사용
brew install ollama

# 또는 공식 installer 다운로드
curl -fsSL https://ollama.ai/install.sh | sh
```

### Linux
```bash
curl -fsSL https://ollama.ai/install.sh | sh
```

## 2. Ollama 서버 시작

```bash
# Ollama 서버 시작 (백그라운드 실행)
ollama serve
```

기본적으로 `http://localhost:11434`에서 서비스됩니다.

## 3. 추천 모델 다운로드

### 경량 모델 (4GB 이하)
```bash
# Llama 3.2 3B - 빠르고 효율적
ollama pull llama3.2

# Qwen2.5 3B - 다국어 지원 우수
ollama pull qwen2.5:3b

# Gemma 2B - 매우 경량
ollama pull gemma2:2b

# Phi-3 Mini - Microsoft 개발, 3.8B
ollama pull phi3
```

### 중급 모델 (8GB 이하)
```bash
# Llama 3.1 8B - 균형잡힌 성능
ollama pull llama3.1:8b

# Qwen2.5 7B - 코딩 및 수학 강화
ollama pull qwen2.5:7b
```

### 고성능 모델 (16GB 이상)
```bash
# Llama 3.1 70B - 최고 성능 (32GB+ RAM 권장)
ollama pull llama3.1:70b
```

## 4. 모델 테스트

```bash
# 모델 목록 확인
ollama list

# 모델과 채팅
ollama run llama3.2
```

## 5. 시스템 요구사항

### 최소 요구사항
- **RAM**: 8GB (3B 모델 기준)
- **디스크**: 5GB 여유 공간
- **CPU**: 현대적인 멀티코어 프로세서

### 권장 사양
- **RAM**: 16GB+ (7B 모델 편안한 실행)
- **GPU**: NVIDIA GPU (CUDA 지원) - 선택사항
- **디스크**: SSD 권장

### 모델별 메모리 사용량
| 모델 크기 | 최소 RAM | 권장 RAM |
|----------|----------|----------|
| 2B       | 4GB      | 8GB      |
| 3B       | 6GB      | 8GB      |
| 7B       | 8GB      | 16GB     |
| 13B      | 16GB     | 32GB     |
| 70B      | 32GB     | 64GB     |

## 6. 성능 최적화

### GPU 사용 (NVIDIA)
```bash
# CUDA 지원 확인
nvidia-smi

# GPU 메모리 제한 설정
export OLLAMA_GPU_MEMORY_FRACTION=0.8
```

### CPU 최적화
```bash
# 사용할 CPU 코어 수 설정
export OLLAMA_NUM_THREADS=4
```

## 7. 문제 해결

### 일반적인 문제들

#### Ollama 서버가 시작되지 않음
```bash
# 프로세스 확인
ps aux | grep ollama

# 포트 사용 확인
netstat -tulpn | grep 11434

# 서버 재시작
pkill ollama
ollama serve
```

#### 모델 다운로드 실패
```bash
# 네트워크 확인
curl -I https://ollama.ai

# 수동 재시도
ollama pull llama3.2 --verbose
```

#### 메모리 부족 오류
- 더 작은 모델 사용 (2B, 3B)
- 다른 프로그램 종료
- 가상 메모리 설정 증가

## 8. 한국어 특화 설정

### 한국어 성능이 좋은 모델들
1. **Qwen2.5** - 다국어 지원 우수
2. **Llama 3.2** - 한국어 지원 개선
3. **Gemma2** - Google 모델, 다국어 지원

### 한국어 프롬프트 예시
```python
system_prompt = """당신은 도움이 되는 한국어 AI 어시스턴트입니다. 
한국어로 자연스럽고 정확한 답변을 제공해주세요."""
```

## 9. RAG 프로젝트에서 사용하기

### 환경 변수 설정
```bash
# .env 파일에 추가
LLM_TYPE=ollama
LLM_MODEL=llama3.2
EMBEDDING_TYPE=sentence-transformer
EMBEDDING_MODEL=paraphrase-multilingual-MiniLM-L12-v2
OLLAMA_BASE_URL=http://localhost:11434
```

### Python 코드 예시
```python
from src.enhanced_rag_pipeline import EnhancedRAGPipeline

# 로컬 LLM 사용 RAG 파이프라인
rag = EnhancedRAGPipeline(
    llm_type="ollama",
    llm_model="llama3.2",
    embedding_type="sentence-transformer",
    embedding_model="paraphrase-multilingual-MiniLM-L12-v2"
)
```

## 10. 모니터링

### 시스템 리소스 모니터링
```bash
# CPU 및 메모리 사용량 확인
htop

# GPU 사용량 확인 (NVIDIA)
nvidia-smi -l 1

# Ollama 로그 확인
journalctl -u ollama -f  # Linux systemd
```

## 11. 보안 및 네트워크

### 외부 접근 허용 (선택사항)
```bash
# 모든 인터페이스에서 접근 허용
export OLLAMA_HOST=0.0.0.0:11434
ollama serve
```

⚠️ **주의**: 외부 접근을 허용할 때는 방화벽 설정을 확인하세요.

## 12. 추가 리소스

- [Ollama 공식 문서](https://github.com/ollama/ollama)
- [Ollama 모델 라이브러리](https://ollama.ai/library)
- [Langchain Ollama 문서](https://python.langchain.com/docs/integrations/llms/ollama)