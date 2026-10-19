# Molecular Data EDA

분자 지문과 연속형 특성을 가진 이진 분류 데이터를 탐색하고, 모델링 전에 확인해야 할 데이터 품질과 검증 분할 문제를 정리한 프로젝트입니다. Python과 RDKit으로 분자 구조, 입력 특성 중복, 라벨 충돌, 골격 중복을 함께 점검합니다.

이 저장소는 분석 코드와 방법을 공유합니다. 원본 데이터, 교재, 발표자료, 개별 분자 구조·라벨이 포함된 결과 표, 실행된 노트북은 배포하지 않습니다. 데이터 없이도 [분석 노트북](notebooks/molecular_eda.ipynb)의 코드와 설명, [주요 발견과 해석](docs/findings.md)을 읽을 수 있습니다.

## 무엇을 분석했나

- 결측치, 무한대, 범위 위반, 완전 중복과 클래스 비율
- 분자량·지용성·합성 용이성·약물 유사성의 분포, 이상치, 효과 크기와 상관관계
- 이진 지문의 활성 비율, 상수·희소·동일값 열과 PCA
- canonical SMILES, 지문 벡터, 전체 숫자 입력 각각의 중복 및 라벨 충돌
- 무작위 층화 분할에서 동일 분자·동일 입력·분자 골격이 겹치는 정도

분석에서 전체 숫자 입력이 같지만 라벨이 다른 그룹과, 학습·검증 간 골격 중복을 확인했습니다. 이를 바탕으로 입력 표현의 한계와 검증 분할 기준을 검토했습니다. 모델 학습이나 예측 성능 측정은 수행하지 않았습니다.

## 구성

| 파일 | 역할 |
| --- | --- |
| `build_eda.py` | 노트북 생성, 실행 및 로컬 HTML 보고서 내보내기 |
| `notebooks/molecular_eda.ipynb` | 실행 출력과 데이터가 없는 분석 노트북 |
| `docs/findings.md` | 핵심 집계 결과, 해석의 한계, 후속 모델링 제안 |
| `requirements-eda.txt` | 검증 환경의 패키지 버전 |

## 실행

Python 3.11 환경에서 확인했습니다. 사용 권한이 있는 CSV를 별도로 준비해야 합니다. 원본 데이터의 다운로드 링크나 샘플 행은 제공하지 않습니다.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-eda.txt
python -m ipykernel install --sys-prefix --name python3 --display-name "Python 3"
python build_eda.py --input /path/to/authorized-data.csv
```

`--input`을 생략하면 `MOLECULAR_EDA_INPUT` 환경 변수를 사용하고, 환경 변수도 없으면 `data/train.csv`를 읽습니다. 입력 파일은 수정하지 않습니다. 실행 결과는 `output/eda/`에만 생성됩니다.

| 로컬 출력 | 내용 |
| --- | --- |
| `molecular_eda.ipynb`, `molecular_eda.html` | 실행 결과를 포함한 노트북과 보고서 |
| `EDA_summary.md`, `metrics.json` | 해당 입력에서 계산한 요약 |
| `tables/`, `figures/` | 상세 점검 표와 시각화 |
| `provenance.json` | 입력 파일명·해시, 환경 버전과 난수 시드 |

노트북에서 직접 실행할 때는 저장소 루트 또는 `notebooks/`를 작업 디렉터리로 사용합니다. 한글 차트에는 AppleGothic, 맑은 고딕 또는 나눔고딕 등 한글 폰트가 필요합니다. 환경에 한글 폰트가 없으면 차트의 한글이 깨질 수 있습니다.

데이터를 읽지 않고 공개용 노트북만 다시 생성할 수도 있습니다.

```bash
python build_eda.py --notebook-only
```

## 입력 전제

| 열 | 형식 및 의미 |
| --- | --- |
| `SMILES` | RDKit으로 해석 가능한 분자 구조 문자열 |
| `ecfp_0` … `ecfp_1023` | ECFP 이진 지문 |
| `fcfp_0` … `fcfp_1023` | FCFP 이진 지문 |
| `ptfp_0` … `ptfp_1023` | 패턴 기반 이진 지문 |
| `MolWt`, `clogp`, `sa_score`, `qed` | 숫자형 분자 특성 |
| `label` | **0 = 독성 있음, 1 = 독성 없음** |

코드는 지문을 `ecfp_`, `fcfp_`, `ptfp_` 접두사로 찾습니다. 이 분석은 각 지문군 1,024개 비트를 전제로 작성했으며 세 지문군과 두 클래스가 모두 있어야 합니다. 데이터가 바뀌면 라벨 의미, 열 구성, 표본 수와 분석 기준을 먼저 검토해야 합니다. 예를 들어 희소 비트는 활성 표본 1~9개, 클래스 간 차이 탐색은 활성·비활성 표본이 각각 50개 이상인 비트로 제한합니다.

## 공개 범위

`.gitignore`는 위의 코드·문서·출력 없는 노트북만 Git에 추가할 수 있도록 구성했습니다. 데이터, 로컬 출력, PDF, 이미지, 가상환경과 임시 파일은 기본적으로 제외됩니다. 로컬 결과에는 입력 데이터의 일부와 개별 분자 정보가 포함되므로 그대로 공개하지 않습니다.

추적 중인 노트북을 직접 실행하면 출력이 변경 사항에 포함될 수 있습니다. 커밋 전 `python build_eda.py --notebook-only`로 공개용 노트북을 다시 만들고 `git diff --cached`에서 내용을 확인하세요. 새 파일을 공개할 때는 내용을 검토한 뒤 `.gitignore`에 해당 파일만 추가합니다.
