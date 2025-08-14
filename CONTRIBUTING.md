# 기여 가이드

RAG 프로젝트에 기여해주셔서 감사합니다! 🎉

## 🤝 기여 방법

### 1. 이슈 리포팅
- 버그 발견 시 [GitHub Issues](https://github.com/blacktop39/rag/issues)에 등록
- 기능 요청은 [GitHub Discussions](https://github.com/blacktop39/rag/discussions) 활용

### 2. 코드 기여
1. 이 저장소를 Fork
2. 새로운 브랜치 생성: `git checkout -b feature/amazing-feature`
3. 변경사항 커밋: `git commit -m 'Add amazing feature'`
4. 브랜치에 Push: `git push origin feature/amazing-feature`
5. Pull Request 생성

## 📋 개발 환경 설정

### 1. 저장소 클론
```bash
git clone https://github.com/your-username/rag.git
cd rag
```

### 2. 의존성 설치
```bash
poetry install
```

### 3. 개발 도구 실행
```bash
# 코드 포맷팅
poetry run black .
poetry run isort .

# 타입 체크
poetry run mypy src/

# 테스트
poetry run pytest

# 린트
poetry run flake8 src/
```

## 📝 코딩 컨벤션

### Python 스타일
- **Black** 코드 포매터 사용
- **isort** import 정렬
- **mypy** 타입 힌트 필수
- **flake8** 코드 품질 검사

### 커밋 메시지
```
feat: 새로운 기능 추가
fix: 버그 수정
docs: 문서 수정
style: 코드 포매팅
refactor: 리팩토링
test: 테스트 추가/수정
chore: 빌드/설정 변경
```

### 브랜치 네이밍
- `feature/기능명`: 새로운 기능
- `fix/버그명`: 버그 수정
- `docs/문서명`: 문서 수정
- `refactor/리팩토링명`: 리팩토링

## 🧪 테스트

### 테스트 실행
```bash
poetry run pytest
```

### 새로운 테스트 작성
- `tests/` 디렉토리에 테스트 파일 추가
- 파일명: `test_*.py`
- 함수명: `test_*`

### 테스트 커버리지
```bash
poetry run pytest --cov=src
```

## 📚 문서화

### 코드 문서화
- 모든 함수/클래스에 docstring 작성
- 타입 힌트 필수
- 복잡한 로직에 주석 추가

### README 업데이트
새로운 기능 추가 시 README.md 업데이트

## 🐛 버그 리포트

버그 리포트 시 다음 정보 포함:

1. **환경 정보**
   - OS (Windows/macOS/Linux)
   - Python 버전
   - Poetry 버전

2. **재현 단계**
   - 구체적인 단계 설명
   - 사용한 명령어
   - 입력 데이터

3. **예상 결과 vs 실제 결과**

4. **에러 메시지**
   ```
   전체 에러 메시지 및 스택 트레이스
   ```

## 💡 기능 요청

새로운 기능 제안 시:

1. **사용 사례** 설명
2. **기대 효과** 명시
3. **구현 아이디어** (있다면)
4. **대안** 고려

## 🔍 코드 리뷰

### Pull Request 체크리스트
- [ ] 코드가 프로젝트 컨벤션을 따름
- [ ] 모든 테스트 통과
- [ ] 새로운 기능에 대한 테스트 추가
- [ ] 문서 업데이트 (필요시)
- [ ] 타입 힌트 추가
- [ ] 커밋 메시지가 명확함

### 리뷰 기준
- **기능성**: 코드가 의도한 대로 작동하는가?
- **가독성**: 코드가 이해하기 쉬운가?
- **성능**: 성능 저하는 없는가?
- **보안**: 보안 이슈는 없는가?
- **일관성**: 프로젝트 스타일과 일치하는가?

## 🏷️ 라벨링

### 이슈 라벨
- `bug`: 버그
- `feature`: 새로운 기능
- `enhancement`: 기능 개선
- `documentation`: 문서
- `good first issue`: 초보자 친화적
- `help wanted`: 도움 필요

### PR 라벨
- `ready for review`: 리뷰 준비 완료
- `work in progress`: 작업 중
- `needs changes`: 수정 필요

## 🎯 기여 아이디어

### 쉬운 기여
- 문서 오타 수정
- 예제 코드 추가
- 번역 (영어 ↔ 한국어)
- 테스트 케이스 추가

### 중급 기여
- 버그 수정
- 성능 최적화
- 새로운 LLM 모델 지원
- UI/UX 개선

### 고급 기여
- 새로운 기능 구현
- 아키텍처 개선
- 벤치마크 도구
- CI/CD 개선

## 📞 연락처

궁금한 점이 있으시면:
- [GitHub Issues](https://github.com/blacktop39/rag/issues)
- [GitHub Discussions](https://github.com/blacktop39/rag/discussions)

## 🙏 감사인사

모든 기여자들에게 진심으로 감사드립니다! 🚀