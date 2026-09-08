# AI Hub 데이터 받아서 학습셋에 넣기

원본 v3.2가 쓴 경적/사이렌/소음 클립(`model_original/dataset_manifest.csv`의
`1.자동차_*`, `N-10_*` 파일들)은 아래 AI Hub 데이터에서 나온 것으로 보입니다.

| 용도 | 데이터셋 | dataSetSn |
|---|---|---|
| 경적·사이렌·주행음 (핵심) | **도시 소리 데이터** | 585 |
| 극한 소음 배경 (선택, robustness↑) | 극한 소음 환경 소리 데이터 | 71376 |

두 데이터 모두 aihub.or.kr에서 **로그인 → 데이터 신청 → 승인**이 먼저 필요합니다
(보통 즉시~1일).

---

## 1. aihubshell 설치 (WSL Ubuntu 안에서)

```bash
curl -o ~/aihubshell https://api.aihub.or.kr/api/aihubshell.do
chmod +x ~/aihubshell
sudo cp ~/aihubshell /usr/bin/aihubshell     # 전역 명령으로 등록
```

## 2. API 키 발급

aihub.or.kr 로그인 → 마이페이지 → **API Key 발급**. 키에 특수문자가 있으면
명령에서 작은따옴표로 감쌉니다.

## 3. 파일 트리 확인 (datasetkey = dataSetSn)

```bash
aihubshell -mode l -datasetkey 585
```

폴더 구조 / 파일명 / 용량 / **filekey**가 출력됩니다. `자동차` 또는 `경적` /
`사이렌` / `주행` 이 들어간 하위 폴더의 filekey를 메모하세요. (전체는 수십 GB일
수 있으니 필요한 부분만 받는 걸 권장.)

## 4. 다운로드

```bash
mkdir -p ~/aihub_raw && cd ~/aihub_raw
# 필요한 filekey만 (콤마로 여러 개):
aihubshell -mode d -datasetkey 585 -filekey 51937,51939 -aihubapikey '발급받은키'
# 또는 전체:
aihubshell -mode d -datasetkey 585 -aihubapikey '발급받은키'
```

압축 병합·해제·삭제는 aihubshell이 자동으로 합니다. 압축 해제용으로 데이터
크기의 2~3배 여유 공간을 두세요.

---

## 5. 클래스별로 정리해서 학습셋에 넣기

`prepare_aihub_data.py`가 받은 파일을 `data/raw/{horn,siren,noise}/`로 복사합니다
(원본은 그대로 둠).

### 방법 A — v3.2와 똑같은 subset만 (가장 안전)

```bash
source env.sh
python prepare_aihub_data.py --src ~/aihub_raw --from-manifest
```

`dataset_manifest.csv`에 이름이 있는 파일만, 그 매니페스트가 지정한 클래스로
복사합니다.

### 방법 B — 더 많이 (폴더/파일명으로 분류)

`-mode l` 출력에서 각 클래스가 어느 경로에 있는지 확인한 뒤:

```bash
python prepare_aihub_data.py --src ~/aihub_raw \
  --by-folder h=경적 h=horn s=사이렌 s=siren n=주행 n=일반 n=noise
```

경로(소문자)에 해당 문자열이 들어간 파일을 그 클래스로 복사합니다. 경적 데이터를
v3.2(218개)보다 많이 확보할수록 좋습니다.

---

## 6. 학습

```bash
source env.sh
python build_features.py --data-root data/raw --copies 10
python train.py --epochs 60
cat artifacts/report.txt          # HORN recall 확인
python apply_to_firmware.py       # 펌웨어에 반영
```
