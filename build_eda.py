"""Create and execute the project EDA notebook, then export a readable HTML report."""
from pathlib import Path
import nbformat as nbf
from nbclient import NotebookClient
from nbconvert import HTMLExporter
import textwrap
import argparse
import os

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'output' / 'eda'
cells = []

def md(s):
    cells.append(nbf.v4.new_markdown_cell(textwrap.dedent(s).strip()))

def code(s):
    cells.append(nbf.v4.new_code_cell(textwrap.dedent(s).strip()))

md('''
# Molecular Data EDA: 분자 특성과 독성 라벨 탐색

분자 데이터의 품질, 클래스별 분포, 이진 지문, 구조 중복과 검증 분할을 점검한다.
분석 코드와 방법을 공유하기 위한 노트북이며 원본 데이터와 실행 출력은 포함하지 않는다.
사용 권한이 있는 데이터를 준비한 뒤 실행한다. 입력 파일은 수정하지 않는다.

**라벨 전제: 0 = 독성 있음 / 1 = 독성 없음.** “독성 비율”은 `label == 0`의 비율이다.
다른 데이터를 사용할 때는 라벨 정의와 스키마부터 확인해야 한다.
결과는 입력 데이터의 라벨에 대한 탐색이며 독성 기전이나 인과관계를 입증하지 않는다.

**읽는 순서:** 기본 구조 → 품질 점검 → 라벨 분포 → 연속형 변수 → 지문 변수 → 구조 중복·검증 분리 → 결론.
표와 그림은 실행 시 계산하고 로컬 `output/eda/`에 저장한다.
''')

md('''
## 1. 실행 환경과 데이터 불러오기

저장소 루트 또는 `notebooks/`에서 실행한다. 기본 입력은 `data/train.csv`이다.
다른 입력은 환경 변수 `MOLECULAR_EDA_INPUT`으로 지정한다.
필요 패키지: pandas, numpy, matplotlib, seaborn, scipy, scikit-learn, rdkit, IPython.
`output/eda/tables`에 상세 집계, `output/eda/figures`에 그림을 저장한다.
''')
code('''
from pathlib import Path
import hashlib, json, os, platform, importlib.metadata
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import font_manager
import seaborn as sns
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from rdkit import Chem, RDLogger
from rdkit.Chem.Scaffolds import MurckoScaffold
from IPython.display import display, Markdown

ROOT = Path.cwd()
if not (ROOT / 'build_eda.py').is_file() and (ROOT.parent / 'build_eda.py').is_file():
    ROOT = ROOT.parent
INPUT = Path(os.environ.get('MOLECULAR_EDA_INPUT', str(ROOT / 'data' / 'train.csv'))).expanduser().resolve()
if not INPUT.is_file():
    raise FileNotFoundError('사용 권한이 있는 CSV를 data/train.csv에 넣거나 MOLECULAR_EDA_INPUT을 지정하세요.')
OUT = ROOT / 'output' / 'eda'
TABLES, FIGURES = OUT / 'tables', OUT / 'figures'
TABLES.mkdir(parents=True, exist_ok=True)
FIGURES.mkdir(parents=True, exist_ok=True)
SEED = 42
sns.set_theme(style='whitegrid', context='notebook')
available_fonts = {f.name for f in font_manager.fontManager.ttflist}
font = next((f for f in ['AppleGothic', 'Apple SD Gothic Neo', 'Malgun Gothic', 'NanumGothic', 'Nanum Myeongjo'] if f in available_fonts), 'DejaVu Sans')
plt.rcParams.update({'font.family': font, 'axes.unicode_minus': False, 'figure.dpi': 120, 'savefig.dpi': 160})
pd.set_option('display.max_columns', 15)
pd.set_option('display.max_rows', 30)
COLORS = {0: '#cc6255', 1: '#317c9b'}
LABELS = {0: '독성 있음 (0)', 1: '독성 없음 (1)'}
def savefig(name):
    plt.tight_layout()
    plt.savefig(FIGURES / f'{name}.png', bbox_inches='tight')
    plt.show()
    plt.close()
def table(frame, name):
    frame.to_csv(TABLES / f'{name}.csv', encoding='utf-8-sig', index=True)
    display(frame.round(4))

df = pd.read_csv(INPUT)
desc = ['MolWt', 'clogp', 'sa_score', 'qed']
groups = {p: [c for c in df if c.startswith(p + '_')] for p in ['ecfp', 'fcfp', 'ptfp']}
fp_cols = sum(groups.values(), [])
numeric_cols = fp_cols + desc
assert set(df.label.unique()) == {0, 1}, '라벨 정의를 재확인해야 합니다.'
toxic = df.label.eq(0).astype('int8')
versions = {p: importlib.metadata.version(p) for p in ['pandas','numpy','matplotlib','seaborn','scipy','scikit-learn','rdkit']}
provenance = {'source': INPUT.name, 'sha256': hashlib.sha256(INPUT.read_bytes()).hexdigest(), 'shape': list(df.shape), 'python': platform.python_version(), 'packages': versions, 'seed': SEED}
(OUT / 'provenance.json').write_text(json.dumps(provenance, ensure_ascii=False, indent=2), encoding='utf-8')
print(f'행 {df.shape[0]:,}개 / 열 {df.shape[1]:,}개 / 예측 입력 숫자 변수 {len(numeric_cols):,}개')
display(df[['SMILES'] + desc + ['label']].head())
print('패키지 버전:', versions)
''')
md('''
| 변수 | 역할 | 해석 |
|---|---|---|
| SMILES | 분자 구조 문자열 | 구조 검증·동일 물질 및 골격 확인에 사용 |
| ecfp_0~1023 | ECFP 1,024개 비트 | 분자 구조 지문 |
| fcfp_0~1023 | FCFP 1,024개 비트 | 기능적 특성에 관한 지문 |
| ptfp_0~1023 | PTFP 1,024개 비트 | 패턴 기반 지문 |
| MolWt | 연속형 | 분자량 |
| clogp | 연속형 | 계산된 분배계수의 로그값 |
| sa_score | 연속형 | 합성 용이성 점수: 1은 쉬움, 10은 어려움 |
| qed | 연속형 | 약물 유사성 점수 |
| label | 예측 목표 | 0은 독성 있음, 1은 독성 없음 |

지문은 숫자형이지만 0/1 존재 여부이므로 연속형 변수와 분리해 탐색한다.
비트 번호만으로 특정 작용기나 독성 기전을 단정할 수 없다.
''')

md('''
## 2. 데이터 품질 점검

결측치, 무한대, 공백 문자열, 완전 중복 행, SMILES 문자열 중복, 범위 위반을 확인한다.
이상치는 다음 절에서 별도로 탐색하며 범위가 크다는 이유만으로 제거하지 않는다.
''')
code('''
Xfp_raw = df[fp_cols].to_numpy()
invalid_fp = ~np.isin(Xfp_raw, [0, 1])
quality = pd.Series({
    '행 수': len(df), '열 수': df.shape[1],
    '결측 셀': int(df.isna().sum().sum()),
    '숫자 무한대 셀': int(np.isinf(df[numeric_cols].to_numpy()).sum()),
    'SMILES 공백/빈 문자열': int(df.SMILES.fillna('').str.strip().eq('').sum()),
    '완전 중복 행(첫 행 제외)': int(df.duplicated().sum()),
    '중복 SMILES 문자열(첫 행 제외)': int(df.SMILES.duplicated().sum()),
    '0/1이 아닌 지문 셀': int(invalid_fp.sum()),
    'MolWt <= 0': int(df.MolWt.le(0).sum()),
    'sa_score 범위 1~10 밖': int((~df.sa_score.between(1,10)).sum()),
    'qed 범위 0~1 밖': int((~df.qed.between(0,1)).sum()),
}, name='값').to_frame()
table(quality, 'quality')
schema = pd.DataFrame({'dtype': df.dtypes.astype(str), 'missing': df.isna().sum(), 'nunique': df.nunique()})
schema.to_csv(TABLES / 'column_profile.csv', encoding='utf-8-sig')
assert not invalid_fp.any(), '0/1 이외의 값이 있어 지문 분석을 중단합니다.'
Xfp = Xfp_raw.astype(np.uint8)
del Xfp_raw, invalid_fp
''')

md('''
## 3. 라벨 분포와 행 순서

다수 클래스만 예측할 때의 정확도를 비교 기준으로 계산한다.
이는 모델 학습 결과가 아니며 높은 정확도만으로 독성 물질을 잘 찾았다고 판단할 수 없다.
행 순서 10개 구간은 시간 구간이 아니다. 파일 정렬에 따른 편향 가능성을 확인하는 용도이다.
''')
code('''
label_table = df.label.value_counts().sort_index().rename('count').to_frame()
label_table['meaning'] = label_table.index.map(LABELS)
label_table['percent'] = label_table['count'] / len(df) * 100
table(label_table, 'label_distribution')
majority_acc = label_table['count'].max() / len(df)
print(f'다수 클래스 기준 정확도: {majority_acc:.2%}')
fig, axes = plt.subplots(1,2, figsize=(12,4))
axes[0].bar([LABELS[0], LABELS[1]], label_table['count'], color=[COLORS[0], COLORS[1]])
for i, n in enumerate(label_table['count']):
    axes[0].text(i, n+80, f'{n:,} ({n/len(df):.1%})', ha='center')
axes[0].set_ylim(0, label_table['count'].max()*1.18)
axes[0].set(title='독성 라벨 분포', ylabel='물질 수')
row_blocks = pd.DataFrame({'block': np.minimum(np.arange(len(df))*10//len(df)+1, 10), 'toxic': toxic})
row_rates = row_blocks.groupby('block').toxic.agg(['size','mean'])
axes[1].plot(row_rates.index, row_rates['mean']*100, marker='o', color=COLORS[0])
axes[1].axhline(toxic.mean()*100, color='gray', linestyle='--', label='전체 독성 비율')
axes[1].set(title='파일 순서별 독성 비율', xlabel='행 순서 구간', ylabel='독성 비율 (%)', ylim=(0,100))
axes[1].legend()
savefig('01_labels_and_row_order')
table(row_rates, 'row_order_blocks')
''')

md('''
## 4. 연속형 변수의 분포·이상치

중앙값·사분위수와 함께 극단값을 확인한다. IQR 기준 이상치는
`Q1 - 1.5×IQR` 미만 또는 `Q3 + 1.5×IQR` 초과 값이다.
이는 통계적 탐색 기준이며 잘못된 측정값이라는 뜻은 아니다.
''')
code('''
table(df[desc].describe(percentiles=[.01,.25,.5,.75,.99]).T, 'descriptor_summary')
by_label = df.groupby('label')[desc].agg(['mean','median','std'])
table(by_label, 'descriptors_by_label')
fig, axes = plt.subplots(2,2, figsize=(12,8))
for c, ax in zip(desc, axes.flat):
    bins = np.histogram_bin_edges(df[c], bins=45)
    for label in [0,1]:
        ax.hist(df.loc[df.label.eq(label),c], bins=bins, density=True, alpha=.45, color=COLORS[label], label=LABELS[label])
    ax.set(title=f'{c} 분포', xlabel=c, ylabel='클래스별 확률밀도')
    ax.legend(fontsize=9)
savefig('02_descriptor_distributions')
fig, axes = plt.subplots(1,4, figsize=(14,4))
for c, ax in zip(desc,axes):
    sns.boxplot(data=df, x='label', y=c, hue='label', order=[0,1], palette=COLORS, legend=False, showfliers=True, fliersize=2, ax=ax)
    ax.set(title=c, xlabel='label (0=독성 있음)', ylabel=c)
savefig('03_descriptor_boxplots')
outlier_rows=[]
outlier_flags = pd.DataFrame(index=df.index)
for c in desc:
    q1,q3=df[c].quantile([.25,.75]); iqr=q3-q1
    lower,upper=q1-1.5*iqr,q3+1.5*iqr
    flag=(df[c]<lower)|(df[c]>upper); outlier_flags[c]=flag
    outlier_rows.append({'feature':c,'lower':lower,'upper':upper,'outlier_count':int(flag.sum()),'outlier_percent':flag.mean()*100})
outlier_table=pd.DataFrame(outlier_rows).set_index('feature')
table(outlier_table,'outlier_summary')
outlier_details=df.loc[outlier_flags.any(axis=1), ['SMILES']+desc+['label']].copy()
outlier_details.insert(0,'csv_row',outlier_details.index+2)
outlier_details['flagged_features']=outlier_flags.loc[outlier_details.index].apply(lambda r:', '.join(r.index[r]),axis=1)
outlier_details.to_csv(TABLES/'outlier_rows.csv',index=False,encoding='utf-8-sig')
print(f'한 개 이상 IQR 범위를 벗어나는 물질: {len(outlier_details):,}개. 원본에서 제거하지 않음.')
display(df.nlargest(5,'MolWt')[['SMILES']+desc+['label']])
''')

md('''
## 5. 독성과의 관계·변수 간 상관관계

표준화 평균 차이는 `(독성 있음 평균 - 독성 없음 평균) / pooled 표준편차`로 계산한다.
양수이면 독성 있음 그룹에서 평균이 크다. 상관계수는 분석용 `toxic = (label == 0)`와 계산한다.
전체 학습 데이터에서 관찰한 관계로, 인과관계나 새 데이터 예측 성능을 의미하지 않는다.
''')
code('''
effects=[]
for c in desc:
    a=df.loc[toxic.eq(1),c]; b=df.loc[toxic.eq(0),c]
    pooled=np.sqrt(((len(a)-1)*a.var()+(len(b)-1)*b.var())/(len(a)+len(b)-2))
    effects.append({'feature':c,'toxic_mean':a.mean(),'non_toxic_mean':b.mean(),'mean_difference':a.mean()-b.mean(),'cohens_d':(a.mean()-b.mean())/pooled,'spearman_with_toxic':spearmanr(df[c],toxic).statistic})
effects=pd.DataFrame(effects).set_index('feature')
effects=effects.loc[effects.cohens_d.abs().sort_values(ascending=False).index]
table(effects,'descriptor_effects')
corr_frame=df[desc].assign(toxic=toxic)
fig,axes=plt.subplots(1,2,figsize=(12,4.5))
for method,ax in zip(['pearson','spearman'],axes):
    corr=corr_frame.corr(method=method)
    corr.to_csv(TABLES/f'correlation_{method}.csv',encoding='utf-8-sig')
    sns.heatmap(corr,annot=True,fmt='.2f',vmin=-1,vmax=1,cmap='RdBu_r',square=True,ax=ax)
    ax.set_title(f'{method.title()} 상관계수 (toxic=1: 독성 있음)')
savefig('04_correlations')

def wilson(k,n):
    # 각 구간의 이항 비율에 대한 Wilson 95% 구간; 다중비교 보정 없음.
    p=k/n; z=1.96; den=1+z*z/n
    center=(p+z*z/(2*n))/den
    half=z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/den
    return center-half,center+half

fig,axes=plt.subplots(2,2,figsize=(12,8))
quantile_tables=[]
for c,ax in zip(desc,axes.flat):
    binned=pd.qcut(df[c],q=5,duplicates='drop')
    agg=pd.DataFrame({'bin':binned,'toxic':toxic}).groupby('bin',observed=True).toxic.agg(['size','sum','mean'])
    low,high=wilson(agg['sum'].to_numpy(),agg['size'].to_numpy())
    agg['ci_low']=low;agg['ci_high']=high;agg['feature']=c
    quantile_tables.append(agg.reset_index())
    ax.errorbar(range(len(agg)),agg['mean']*100,yerr=np.vstack([agg['mean']-low,high-agg['mean']])*100,fmt='o-',capsize=4,color=COLORS[0])
    ax.axhline(toxic.mean()*100,linestyle='--',color='gray')
    ax.set_xticks(range(len(agg)),[f'Q{i+1}\\nn={n:,}' for i,n in enumerate(agg['size'])])
    ax.set(title=f'{c}: 낮은 값부터 5분위',ylabel='독성 비율 (%)',ylim=(0,100))
savefig('05_quantiles_toxicity')
quantiles=pd.concat(quantile_tables,ignore_index=True)
table(quantiles,'descriptor_quantiles')
''')

md('''
## 6. 분자 지문: 활성 비율·상수·중복

활성 비율은 각 비트가 1인 물질의 비율이다. 전체가 0 또는 1인 상수 열은 현재 표본에서 구분 정보를 주지 않는다.
희소 비트는 1인 물질이 1~9개인 열로 정의한다. 중복 열은 **현재 표본에서만** 값이 완전히 같다는 의미이다.
상수·희소·중복 열을 실제 모델에서 제거할 때는 학습 분할 안에서 기준을 정해야 한다.
''')
code('''
activation=df[fp_cols].mean()
counts=df[fp_cols].sum()
# 0/1 열을 비트로 압축한 뒤 바이트로 비교하여, 큰 전치 DataFrame 없이 정확히 중복 확인.
packed=np.packbits(Xfp,axis=0)
seen={}; duplicate_pairs=[]
for i,c in enumerate(fp_cols):
    key=packed[:,i].tobytes()
    if key in seen:
        duplicate_pairs.append({'duplicate':c,'representative':seen[key],'constant':bool(activation[c] in [0,1])})
    else:
        seen[key]=c
duplicate_table=pd.DataFrame(duplicate_pairs,columns=['duplicate','representative','constant'])
duplicate_table.to_csv(TABLES/'duplicate_fingerprint_columns.csv',index=False,encoding='utf-8-sig')
fp_summary=[]; active_counts=pd.DataFrame(index=df.index)
for g,cols in groups.items():
    a=activation[cols]
    active_counts[g]=df[cols].sum(axis=1)
    fp_summary.append({'family':g,'columns':len(cols),'ones_percent':a.mean()*100,'zero_percent':(1-a.mean())*100,'constant_zero':int(a.eq(0).sum()),'constant_one':int(a.eq(1).sum()),'rare_1_to_9':int(counts[cols].between(1,9).sum()),'mean_active_per_molecule':active_counts[g].mean(),'all_zero_rows':int(active_counts[g].eq(0).sum())})
fp_summary=pd.DataFrame(fp_summary).set_index('family')
table(fp_summary,'fingerprint_summary')
print(f'전체 지문에서 첫 열을 제외한 동일값 중복 열: {len(duplicate_table):,}개 (상수 열 포함)')
print(f'이 중 비상수 중복 열: {int((~duplicate_table.constant.astype(bool)).sum()):,}개')
pd.DataFrame({'active_count':counts,'activation_rate':activation}).to_csv(TABLES/'fingerprint_activation.csv',encoding='utf-8-sig')
fig,axes=plt.subplots(1,3,figsize=(14,4))
for (g,cols),ax in zip(groups.items(),axes):
    ax.hist(activation[cols]*100,bins=np.linspace(0,100,41),color='#317c9b')
    ax.set(title=f'{g.upper()} 비트 활성 비율',xlabel='1인 물질의 비율 (%)',ylabel='비트 수')
savefig('06_fingerprint_activation')
fig,axes=plt.subplots(1,3,figsize=(14,4))
for g,ax in zip(groups,axes):
    for label in [0,1]:
        ax.hist(active_counts.loc[df.label.eq(label),g],bins=35,density=True,alpha=.45,color=COLORS[label],label=LABELS[label])
    ax.set(title=f'{g.upper()} 물질별 활성 비트 수',xlabel='1인 비트 수',ylabel='클래스별 확률밀도')
    ax.legend(fontsize=9)
savefig('07_active_bits_per_molecule')
table(active_counts.assign(label=df.label).groupby('label').agg(['mean','median','std']),'active_bits_by_label')
''')

md('''
## 7. 독성별 지문 차이와 2차원 탐색

`P(비트=1 | 독성 있음) - P(비트=1 | 독성 없음)`을 계산한다.
비트가 1인 물질과 0인 물질이 각각 50개 이상인 경우만 후보로 보고
동일한 값을 갖는 열은 대표 열 하나만 남겨 차이가 큰 15개를 표시한다.
여러 비트 중 큰 차이를 골라 본 탐색 결과이며 검증된 중요도 또는 원인으로 해석하지 않는다.

PCA는 상수 열을 제외한 전체 지문을 평균 중심화하고 2차원으로 축약한다.
비트별 표준화는 하지 않는다. 라벨은 좌표 계산에 쓰지 않고 색상에만 사용한다.
전체 표본을 이용한 시각화이므로 모델 검증을 대신하지 않는다.
''')
code('''
rate_toxic=df.loc[toxic.eq(1),fp_cols].mean()
rate_non=df.loc[toxic.eq(0),fp_cols].mean()
bit_effect=pd.DataFrame({'active_count':counts,'toxic_activation':rate_toxic,'non_toxic_activation':rate_non,'difference_pp':(rate_toxic-rate_non)*100})
bit_effect['eligible']=counts.ge(50)&counts.le(len(df)-50)
bit_effect['duplicate']=bit_effect.index.isin(duplicate_table['duplicate'])
bit_effect.to_csv(TABLES/'fingerprint_label_differences.csv',encoding='utf-8-sig')
ranked=bit_effect.loc[bit_effect.eligible & ~bit_effect.duplicate].copy()
top_bits=ranked.loc[ranked.difference_pp.abs().nlargest(15).index].sort_values('difference_pp')
fig,ax=plt.subplots(figsize=(9,6))
ax.barh(top_bits.index,top_bits.difference_pp,color=[COLORS[0] if v>0 else COLORS[1] for v in top_bits.difference_pp])
ax.axvline(0,color='gray',linewidth=1)
ax.set(title='독성 그룹 간 활성 비율 차이가 큰 지문 15개',xlabel='독성 있음 - 독성 없음 활성 비율 (%p)')
savefig('08_fingerprint_label_differences')
table(top_bits[['active_count','toxic_activation','non_toxic_activation','difference_pp']],'top_fingerprint_differences')
nonconstant=(activation.gt(0)&activation.lt(1)).to_numpy()
pca=PCA(n_components=2,svd_solver='randomized',random_state=SEED)
coords=pca.fit_transform(Xfp[:,nonconstant].astype(np.float32))
order=np.random.default_rng(SEED).permutation(len(df))
fig,ax=plt.subplots(figsize=(8,6))
ax.scatter(coords[order,0],coords[order,1],c=[COLORS[int(x)] for x in df.label.iloc[order]],s=7,alpha=.3,rasterized=True)
from matplotlib.lines import Line2D
ax.legend(handles=[Line2D([0],[0],marker='o',color='w',markerfacecolor=COLORS[k],label=LABELS[k],markersize=8) for k in [0,1]])
ax.set(title=f'지문 PCA: 설명 분산 합 {pca.explained_variance_ratio_.sum():.1%}',xlabel=f'PC1 ({pca.explained_variance_ratio_[0]:.1%})',ylabel=f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
savefig('09_fingerprint_pca')
pd.DataFrame(coords,columns=['PC1','PC2']).assign(label=df.label).to_csv(TABLES/'fingerprint_pca.csv',index=False,encoding='utf-8-sig')
''')

md('''
## 8. SMILES 구조 검증·동일 분자·골격 중복

RDKit으로 파싱하고 입체정보를 포함한 canonical SMILES로 정규화한다.
문자열이 다르더라도 같은 분자로 정규화되는지, 같은 분자의 라벨이 충돌하는지 확인한다.
염 제거, tautomer 통일, 전하 중성화는 수행하지 않으므로 이러한 차이는 남을 수 있다.

Bemis–Murcko 골격은 입체정보 없이 계산한다. 빈 골격(고리 없는 분자)은 하나의 거대 그룹으로 묶지 않는다.
구조가 비슷한 물질이 학습·검증에 동시에 들어갈 가능성을 살펴보기 위해
80:20 라벨 층화 무작위 분할을 **진단용으로만** 만든다. 모델은 학습하지 않는다.
''')
code('''
RDLogger.DisableLog('rdApp.*')
mols=[Chem.MolFromSmiles(s) for s in df.SMILES]
canonical=pd.Series([Chem.MolToSmiles(m,isomericSmiles=True) if m is not None else None for m in mols],index=df.index)
scaffold=pd.Series([MurckoScaffold.MurckoScaffoldSmiles(mol=m,includeChirality=False) if m is not None else None for m in mols],index=df.index)
structure=pd.DataFrame({'csv_row':df.index+2,'SMILES':df.SMILES,'canonical_smiles':canonical,'scaffold':scaffold,'label':df.label})
canonical_labels=structure.dropna(subset=['canonical_smiles']).groupby('canonical_smiles').label.nunique()
conflicts=canonical_labels[canonical_labels.gt(1)]
structure.to_csv(TABLES/'molecule_structure_audit.csv',index=False,encoding='utf-8-sig')
structure.loc[canonical.duplicated(keep=False)&canonical.notna()].to_csv(TABLES/'canonical_duplicate_rows.csv',index=False,encoding='utf-8-sig')
structure.loc[canonical.isin(conflicts.index)].to_csv(TABLES/'conflicting_label_rows.csv',index=False,encoding='utf-8-sig')
# 정확히 같은 지문 벡터도 확인. 서로 다른 분자가 동일 지문을 가질 수 있다.
fp_keys=pd.Series([r.tobytes() for r in np.packbits(Xfp,axis=1)])
fp_group_labels=pd.DataFrame({'fp':fp_keys,'label':df.label}).groupby('fp').label.nunique()
full_keys=pd.Series([a+b.tobytes() for a,b in zip(fp_keys,df[desc].to_numpy(dtype=np.float64))])
full_group_labels=pd.DataFrame({'features':full_keys,'label':df.label}).groupby('features').label.nunique()
conflicting_feature_keys=set(full_group_labels[full_group_labels.gt(1)].index)
feature_audit=structure.copy()
feature_audit['feature_group_id']=pd.factorize(full_keys,sort=False)[0]
feature_audit['feature_label_conflict']=full_keys.isin(conflicting_feature_keys)
for c in desc:
    feature_audit[c]=df[c]
feature_audit.loc[full_keys.duplicated(keep=False)].to_csv(TABLES/'identical_numeric_feature_rows.csv',index=False,encoding='utf-8-sig')
feature_audit.loc[feature_audit.feature_label_conflict].to_csv(TABLES/'numeric_feature_label_conflicts.csv',index=False,encoding='utf-8-sig')
scaffold_valid=scaffold.notna()&scaffold.ne('')
scaffold_sizes=scaffold[scaffold_valid].value_counts()
structure_summary=pd.Series({
    'SMILES 파싱 실패':int(canonical.isna().sum()),
    'canonical 중복 행(첫 행 제외)':int(canonical.dropna().duplicated().sum()),
    '동일 canonical 라벨 충돌 그룹':len(conflicts),
    '동일 전체 지문 중복 행(첫 행 제외)':int(fp_keys.duplicated().sum()),
    '동일 전체 지문 라벨 충돌 그룹':int(fp_group_labels.gt(1).sum()),
    '동일 전체 숫자 입력 중복 행(첫 행 제외)':int(full_keys.duplicated().sum()),
    '동일 전체 숫자 입력 라벨 충돌 그룹':int(full_group_labels.gt(1).sum()),
    '고유 비어 있지 않은 골격':len(scaffold_sizes),
    '빈 골격 물질':int(scaffold.eq('').sum()),
    '2개 이상 물질이 공유하는 골격':int(scaffold_sizes.ge(2).sum()),
    '공유 골격에 속한 물질':int(scaffold_sizes[scaffold_sizes.ge(2)].sum()),
    '최대 골격 그룹 크기':int(scaffold_sizes.max()) if len(scaffold_sizes) else 0,
},name='값').to_frame()
table(structure_summary,'structure_summary')
train_idx,val_idx=train_test_split(df.index,test_size=.2,random_state=SEED,stratify=df.label)
train_scaffolds=set(scaffold.loc[train_idx].dropna())-{''}
val_valid=scaffold.loc[val_idx].notna()&scaffold.loc[val_idx].ne('')
seen_scaffold=scaffold.loc[val_idx].isin(train_scaffolds)&val_valid
overlap_rate=seen_scaffold.sum()/val_valid.sum() if val_valid.sum() else np.nan
canonical_overlap=canonical.loc[val_idx].isin(set(canonical.loc[train_idx].dropna()))&canonical.loc[val_idx].notna()
feature_overlap=full_keys.loc[val_idx].isin(set(full_keys.loc[train_idx]))
display(Markdown(f'진단용 무작위 분할: 학습 **{len(train_idx):,}개**, 검증 **{len(val_idx):,}개**. '
                 f'검증의 비어 있지 않은 골격 물질 {val_valid.sum():,}개 중 **{seen_scaffold.sum():,}개 ({overlap_rate:.1%})**가 '
                 f'학습에 존재하는 골격을 갖는다. 동일 canonical 분자 중복은 **{canonical_overlap.sum():,}개**, '
                 f'동일 숫자 입력 중복은 **{feature_overlap.sum():,}개**이다.'))
display(Markdown(f'**같은 입력, 다른 라벨:** 지문과 연속형 4개를 모두 포함한 숫자 입력이 동일한데 라벨이 다른 그룹이 '
                 f'**{int(full_group_labels.gt(1).sum()):,}개**이다. 상세 목록은 `tables/numeric_feature_label_conflicts.csv`에 저장했다. '
                 '같은 canonical 분자의 라벨 충돌과는 구분해야 한다. 지문이 일부 구조 차이를 표현하지 못했거나 '
                 '제공되지 않은 실험 조건 등이 다를 가능성이 있지만, 현재 자료만으로 원인을 확정할 수 없다. '
                 '라벨 수정·다수결·자동 삭제는 하지 않았다.'))
fig,ax=plt.subplots(figsize=(10,4))
top_scaffold=scaffold_sizes.head(15)
ax.bar(range(len(top_scaffold)),top_scaffold.to_numpy(),color='#317c9b')
ax.set_xticks(range(len(top_scaffold)),[str(i+1) for i in range(len(top_scaffold))])
ax.set(title='물질 수가 많은 골격 상위 15개',xlabel='골격 순위 (구조 문자열은 상세 표 참조)',ylabel='물질 수')
savefig('10_scaffold_sizes')
table(top_scaffold.rename('count').rename_axis('scaffold').to_frame(),'top_scaffolds')
''')

md('''
## 9. 결론과 모델링으로 넘길 사항
''')
code('''
clogp_quintiles=quantiles.loc[quantiles.feature.eq('clogp')]
constant_count=int((activation.eq(0)|activation.eq(1)).sum())
summary=f"""### 이번 데이터에서 확인한 핵심 결과

- **데이터:** {len(df):,}행 × {df.shape[1]:,}열. 숫자 입력은 지문 {len(fp_cols):,}개와 연속형 4개이다.
- **라벨:** 독성 있음 {int(toxic.sum()):,}개 ({toxic.mean():.2%}), 독성 없음 {int((1-toxic).sum()):,}개 ({(1-toxic).mean():.2%}). 다수 클래스 정확도 기준은 {majority_acc:.2%}이다.
- **품질:** 결측 셀 {int(df.isna().sum().sum()):,}개, 완전 중복 행 {int(df.duplicated().sum()):,}개, SMILES 파싱 실패 {int(canonical.isna().sum()):,}개. canonical 중복은 첫 행 제외 {int(canonical.dropna().duplicated().sum()):,}개이다.
- **clogp:** 독성 있음 평균 {df.loc[toxic.eq(1),'clogp'].mean():.3f}, 독성 없음 평균 {df.loc[toxic.eq(0),'clogp'].mean():.3f}. 표준화 평균 차이는 {effects.loc['clogp','cohens_d']:.3f}이다. 가장 낮은 5분위의 독성 비율은 {clogp_quintiles.iloc[0]['mean']:.1%}, 가장 높은 5분위는 {clogp_quintiles.iloc[-1]['mean']:.1%}이다.
- **MolWt:** 독성 있음 평균 {df.loc[toxic.eq(1),'MolWt'].mean():.2f}, 독성 없음 평균 {df.loc[toxic.eq(0),'MolWt'].mean():.2f}. 평균 차이는 작으며 최댓값 {df.MolWt.max():.3f}을 포함한 극단값은 개별 검토 대상으로 남겼다.
- **지문:** 상수 열 {constant_count:,}개. 현재 표본에서 동일값인 중복 열은 첫 열 제외 {len(duplicate_table):,}개이며 이 중 비상수 열은 {int((~duplicate_table.constant.astype(bool)).sum()):,}개이다. 이 두 수는 중복될 수 있어 더하지 않는다.
- **표현의 한계:** 전체 지문이 같은 중복 행은 첫 행 제외 {int(fp_keys.duplicated().sum()):,}개이며 그중 서로 다른 라벨을 포함하는 그룹은 {int(fp_group_labels.gt(1).sum()):,}개이다. 연속형 4개까지 포함한 전체 숫자 입력으로 비교해도 중복 {int(full_keys.duplicated().sum()):,}개와 라벨 충돌 그룹 **{int(full_group_labels.gt(1).sum()):,}개**가 남는다. 같은 canonical 분자의 라벨 충돌은 {len(conflicts):,}개이므로 이를 원본 라벨 오류로 단정하지 않는다.
- **변수 관계:** MolWt와 qed의 Pearson 상관계수는 {df[['MolWt','qed']].corr().iloc[0,1]:.3f}이다. ECFP와 FCFP의 0 비율은 각각 {fp_summary.loc['ecfp','zero_percent']:.1f}%, {fp_summary.loc['fcfp','zero_percent']:.1f}%로 높고 PTFP는 {fp_summary.loc['ptfp','zero_percent']:.1f}%이다. 같은 척도라도 지문별 분포가 다르다.
- **차원 축소:** PCA 두 축의 설명 분산 합은 {pca.explained_variance_ratio_.sum():.1%}이다. 2차원에서 색이 겹치는 모습만으로 원래 고차원 공간의 분류 가능성을 판단할 수 없다.
- **검증 분리:** 무작위 80:20 분할의 검증 물질 중 비어 있지 않은 골격 기준 {overlap_rate:.1%}가 학습과 골격을 공유한다. 구조가 새로운 물질로 일반화하는 성능을 보려면 별도의 골격 그룹 분할을 함께 평가할 필요가 있다.

### 전처리·검증 제안

1. 라벨은 원본 그대로 유지한다. `label=1` 기준 F1과 독성 탐지 목적의 `label=0` 재현율을 구분해서 기록한다. F1의 양성 클래스와 평균 방식도 함께 기록한다.
2. 모델링 전에 검증 세트를 분리하고 고정한다. 동일 canonical 분자와 동일 숫자 입력이 분할을 가로지르지 않도록 그룹을 구성한다. 무작위 층화 분할 외에 골격 그룹 분할도 비교하되 골격 분할에서는 클래스 비율과 그룹 크기를 확인한다. 서로 다른 중복 기준이 연결되는 행은 같은 그룹으로 묶고 분할 후 중복 여부를 다시 확인한다.
3. 결측치·범위 위반이 없다면 불필요한 대체를 하지 않는다. IQR 이상치는 자동 삭제하지 말고 원본 및 구조를 확인한다.
4. 상수·중복·희소 지문 제거 기준과 스케일러는 학습 분할에서만 학습한다. 연속형 4개는 모델에 따라 표준화하고 이진 지문은 별도로 처리한다. 전체 EDA에서 선택한 상위 비트를 그대로 확정하면 검증 라벨 정보가 선택에 개입할 수 있다.
5. ECFP, FCFP, PTFP 각각과 조합을 같은 분할에서 비교한다. 상관관계와 PCA 시각화만으로 모델의 유효성 또는 지문 우열을 판정하지 않는다.
6. 이번 EDA는 전체 입력 데이터를 보았으므로 이후 내부 분할의 성능은 탐색적 비교로 해석한다. 완전히 독립적인 최종 평가는 미열람 외부 데이터가 필요하다.
7. 동일 숫자 입력의 라벨 충돌은 `tables/numeric_feature_label_conflicts.csv`로 검토한다. 원인을 확인하기 전에는 임의로 라벨을 합치거나 바꾸지 않는다. 현재 숫자 입력만 사용하는 결정적 분류기는 이 행들을 모두 맞힐 수 없다.

### 지표 해석 시 주의

`label=1`을 Positive로 둘 때 독성 탐지 재현율은 `recall_score(y_true, y_pred, pos_label=0)`이다.
이는 해당 Positive 정의에서 **Specificity = TN / (TN + FP)** 와 같다.
**NPV = TN / (TN + FN)** 는 음성 예측 중 실제 음성의 비율이다. Specificity와 구분해서 기록한다.
ROC-AUC와 PR-AUC도 어느 클래스의 확률을 사용했는지 함께 명시한다.

본 작업은 EDA이며 예측 모델의 학습·튜닝·제출 파일 작성은 포함하지 않았다.
"""
display(Markdown(summary))
(OUT/'EDA_summary.md').write_text('# Molecular Data EDA 요약\\n\\n'+summary+'\\n\\n입력 CSV에서 계산한 로컬 실행 결과. 상세 계산은 molecular_eda.ipynb 참조.\\n',encoding='utf-8')
metrics={'rows':len(df),'columns':df.shape[1],'toxic_count':int(toxic.sum()),'non_toxic_count':int((1-toxic).sum()),'constant_fp_columns':constant_count,'duplicate_fp_columns':len(duplicate_table),'canonical_duplicate_rows':int(canonical.dropna().duplicated().sum()),'canonical_conflict_groups':len(conflicts),'numeric_input_duplicate_rows':int(full_keys.duplicated().sum()),'numeric_input_conflict_groups':int(full_group_labels.gt(1).sum()),'validation_numeric_input_overlap':int(feature_overlap.sum()),'scaffold_overlap_rate':float(overlap_rate),'pca_explained_variance':pca.explained_variance_ratio_.tolist()}
(OUT/'metrics.json').write_text(json.dumps(metrics,ensure_ascii=False,indent=2),encoding='utf-8')
''')

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, help='CSV path; defaults to data/train.csv')
    parser.add_argument('--notebook-only', action='store_true', help='Generate a clean notebook without reading data')
    args = parser.parse_args()
    nb = nbf.v4.new_notebook(cells=cells)
    nb.metadata = {
        'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
        'language_info': {'name': 'python'},
    }
    # Stable cell IDs keep regenerated, unexecuted notebooks reviewable.
    for i, cell in enumerate(nb.cells):
        cell.id = f'eda-{i:03d}'
    if args.notebook_only:
        path = ROOT / 'notebooks' / 'molecular_eda.ipynb'
        path.parent.mkdir(parents=True, exist_ok=True)
        nbf.write(nb, path)
        print('Created clean notebook:', path)
        return
    input_path = (args.input or Path(os.environ.get('MOLECULAR_EDA_INPUT', str(ROOT / 'data' / 'train.csv')))).expanduser().resolve()
    if not input_path.is_file():
        parser.error('Input CSV not found. Supply --input or place an authorized CSV at data/train.csv.')
    OUT.mkdir(parents=True, exist_ok=True)
    client = NotebookClient(nb, timeout=600, kernel_name='python3', resources={'metadata': {'path': str(ROOT)}})
    client.execute(env={**os.environ, 'MOLECULAR_EDA_INPUT': str(input_path)})
    nbf.write(nb, OUT / 'molecular_eda.ipynb')
    exporter = HTMLExporter(template_name='lab')
    exporter.exclude_input = True
    exporter.exclude_input_prompt = True
    exporter.exclude_output_prompt = True
    body, _ = exporter.from_notebook_node(nb)
    (OUT / 'molecular_eda.html').write_text(body, encoding='utf-8')
    print('Created local executed notebook and HTML:', OUT)


if __name__ == '__main__':
    main()
