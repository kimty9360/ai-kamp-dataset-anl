# RG3 집중 개선 및 Recall 95% 목표 평가

CN7은 기존 좌우 CatBoost balanced를 유지하고 정책만 비교했다. RG3는 XGBoost balanced와 RBF SVM에서 원단위 입력, 물리 차이 변수, 작은 튜닝, 가중치 변경을 각각 고정 비교했다. 기존 외부/내부 그룹 분할을 유지했다. 입력은 원단위로 복구했으나 변수 수와 행은 그대로이며 새로운 결측/이상치 삭제는 하지 않았다.

RG3는 LH 591개 모두 양품, RH는 양품566·불량25개다. 불량 사유는 가스17·미성형8이다. 생산일은 10월21일 불량0, 22일15, 23일10이다. Reason과 시간은 진단용이며 모델 입력에서 제외했다. RH만의 단일 변수 분포/AUC도 전체 라벨 기반 탐색이므로 검증 성능이 아니다.

inner_recall95는 내부 OOF Recall>=.95를 만족하는 임계값 중 Precision 최대, 동률이면 높은 임계값을 고른다. inner_selected는 같은 내부 기준으로 후보를 선택한다. 외부 목표 달성은 보장되지 않는다. 외부 점수의 가장 좋은 후보를 사후 선택한 결과와 구분해야 한다.

|데이터|모델|설정|정책|AP|Precision|Recall|F1|검사비율|Recall@10%|95%달성 폴드|
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|
|cn7|catboost_balanced|reference|inner_f1|0.5541|0.293|0.632|0.312|0.031|0.900|2/9|
|cn7|catboost_balanced|reference|inner_recall95|0.5541|0.052|0.922|0.097|0.340|0.900|6/9|
|rg3|svm_rbf|inner_selected|inner_f1|0.0669|0.038|0.264|0.061|0.120|0.229|1/9|
|rg3|svm_rbf|inner_selected|inner_recall95|0.0669|0.039|0.898|0.074|0.506|0.229|4/9|
|rg3|svm_rbf|raw|inner_f1|0.0709|0.050|0.264|0.066|0.115|0.243|1/9|
|rg3|svm_rbf|raw|inner_recall95|0.0709|0.037|0.899|0.071|0.531|0.243|4/9|
|rg3|svm_rbf|raw_engineered|inner_f1|0.0670|0.037|0.314|0.060|0.153|0.214|1/9|
|rg3|svm_rbf|raw_engineered|inner_recall95|0.0670|0.037|0.946|0.071|0.540|0.214|6/9|
|rg3|svm_rbf|raw_engineered_tuned|inner_f1|0.0670|0.037|0.314|0.060|0.153|0.214|1/9|
|rg3|svm_rbf|raw_engineered_tuned|inner_recall95|0.0670|0.037|0.946|0.071|0.541|0.214|6/9|
|rg3|svm_rbf|raw_engineered_weight|inner_f1|0.0414|0.019|0.212|0.033|0.143|0.142|1/9|
|rg3|svm_rbf|raw_engineered_weight|inner_recall95|0.0414|0.024|0.928|0.047|0.826|0.142|6/9|
|rg3|svm_rbf|reference|inner_f1|0.0709|0.050|0.264|0.066|0.115|0.243|1/9|
|rg3|svm_rbf|reference|inner_recall95|0.0709|0.037|0.899|0.071|0.531|0.243|4/9|
|rg3|xgboost_balanced|inner_selected|inner_f1|0.0541|0.034|0.416|0.060|0.234|0.172|1/9|
|rg3|xgboost_balanced|inner_selected|inner_recall95|0.0541|0.029|0.944|0.057|0.682|0.172|7/9|
|rg3|xgboost_balanced|raw|inner_f1|0.0753|0.039|0.306|0.061|0.187|0.260|1/9|
|rg3|xgboost_balanced|raw|inner_recall95|0.0753|0.027|1.000|0.053|0.793|0.260|9/9|
|rg3|xgboost_balanced|raw_engineered|inner_f1|0.0599|0.033|0.407|0.058|0.236|0.214|0/9|
|rg3|xgboost_balanced|raw_engineered|inner_recall95|0.0599|0.028|0.975|0.054|0.752|0.214|8/9|
|rg3|xgboost_balanced|raw_engineered_tuned|inner_f1|0.0511|0.035|0.457|0.063|0.263|0.144|1/9|
|rg3|xgboost_balanced|raw_engineered_tuned|inner_recall95|0.0511|0.029|0.968|0.056|0.708|0.144|8/9|
|rg3|xgboost_balanced|raw_engineered_weight|inner_f1|0.0604|0.027|0.307|0.045|0.224|0.199|1/9|
|rg3|xgboost_balanced|raw_engineered_weight|inner_recall95|0.0604|0.028|1.000|0.054|0.765|0.199|9/9|
|rg3|xgboost_balanced|reference|inner_f1|0.0753|0.039|0.306|0.061|0.187|0.260|1/9|
|rg3|xgboost_balanced|reference|inner_recall95|0.0753|0.027|1.000|0.053|0.793|0.260|9/9|

summary.csvには固定0.5政策とRecall@5/10/20%も保存した。repeat_counts.csvは各seedで原本行を一回ずつ数えたTP/FP/FN/TN。9fold平均と全行集計は異なる。繰り返しで同じ不良を複数の独立不良と数えない。

候補は事前に固定したが既存EDAを利用した開発比較であり独立テストではない。学習foldだけで前処理・重み・モデルをfitした。新たな生産データは追加していない。時間順検証や未来性能の保証はしていない。原単位差も実際のセンサ定義の限界がある。
