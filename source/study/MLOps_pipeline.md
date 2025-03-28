project-root/
├── .github/workflows/          # CI/CDパイプライン (GitHub Actions)
│   ├── ci.yml                  # Lint, Format Check, Type Check, Test (Unit/Integration), Coverage, MLproject Validate
│   └── cd.yml                  # (Optional) Trigger on merge/tag: Evaluate, Register Model, Deploy API/Jobs
├── .devcontainer/              # (Optional) VS Code Remote - Containers 設定
│   ├── devcontainer.json
│   └── Dockerfile              # 開発用コンテナ定義
├── .env.example                # 環境変数テンプレート (コピーして .env を作成)
├── .gitignore                  # Git管理対象外 (logs/, .env, .mypy_cache, .pytest_cache, __pycache__, etc.)
├── Dockerfile                  # 本番/実行用コンテナイメージ定義
├── LICENSE                     # プロジェクトライセンス
├── MLproject                   # MLflow Project定義 (エントリーポイント, パラメータ)
├── mkdocs.yml                  # (Optional) MkDocs 設定ファイル
├── pyproject.toml              # Poetry/PDM設定, 依存関係, ビルド設定, ツール設定 (ruff, black, isort, mypy, pytest)
├── README.md                   # プロジェクト概要, 開発セットアップ, 主要コマンド, ドキュメントリンク
├── configs/                    # 設定ファイル (アプリケーションレベル)
│   ├── settings.yaml           # 基本設定 (デフォルト値, MLflow設定含む)
│   ├── logging.yaml            # ロギング設定
│   └── model_params/           # (Optional) モデルのハイパーパラメータ設定例
│       ├── sklearn_defaults.yaml
│       └── xgboost_defaults.yaml
├── data/                       # (gitignore推奨) ローカルデータ (主に開発/テスト用)
│   ├── raw/                    # 生データ
│   ├── interim/                # 中間生成データ
│   ├── processed/              # 最終的な処理済みデータ
│   └── sample/                 # サンプルデータ
├── docs/                       # ドキュメントソース (MkDocs/Sphinx用)
│   ├── index.md                # ドキュメントトップ
│   ├── architecture.md
│   ├── api.md                  # API仕様 (OpenAPI参照など)
│   ├── data_pipeline.md
│   ├── feature_engineering.md
│   ├── model_development.md    # MLflow利用方法含む
│   ├── deployment.md
│   ├── contributing.md
│   ├── code_of_conduct.md
│   ├── style_guide.md          # コーディング規約詳細
│   └── release_notes/
│       └── v1.0.0.md
├── notebooks/                  # 実験・分析用Jupyterノートブック (バージョン管理推奨)
│   ├── 01_data_exploration.ipynb
│   ├── 02_feature_prototyping.ipynb
│   ├── 03_model_experimentation.ipynb # MLflow Tracking利用
│   └── 04_result_analysis.ipynb
├── scripts/                    # 補助スクリプト (Makefileで管理することも多い)
│   ├── setup.sh                # 開発環境セットアップ (Poetry/PDM installなど)
│   ├── lint.sh                 # リンター実行 (ruff check .)
│   ├── format.sh               # フォーマッター実行 (black ., isort .)
│   ├── type_check.sh           # 型チェック実行 (mypy src tests)
│   ├── run_tests.sh            # テスト実行 (pytest)
│   └── run_mlflow_local.sh     # MLflow Projectローカル実行例
├── src/                        # Pythonソースコードパッケージ
│   └── my_project/             # パッケージ名 (例: my_project)
│       ├── __init__.py
│       ├── api/                # API層 (FastAPI)
│       │   ├── __init__.py
│       │   ├── main.py         # FastAPI App, MLflowモデルロード (on_event startup)
│       │   ├── dependencies.py # 設定注入, MLflowクライアント, DBセッション等
│       │   ├── routers/        # APIエンドポイント (predict, ingest, status等)
│       │   │   └── ...
│       │   └── schemas/        # Pydanticスキーマ (Request/Response)
│       │       └── ...
│       ├── core/               # コア機能, 共通設定
│       │   ├── __init__.py
│       │   ├── config.py       # Pydantic Settingsによる設定管理
│       │   ├── logging_config.py # ロギング設定読み込み・適用
│       │   └── exceptions.py   # カスタム例外
│       ├── data_processing/    # データ処理パイプラインステップ
│       │   ├── __init__.py
│       │   ├── ingest.py       # データ取り込み
│       │   ├── validate.py     # データ検証
│       │   └── transform.py    # データ変換
│       ├── feature_engineering/ # 特徴量エンジニアリング
│       │   ├── __init__.py
│       │   ├── pipeline.py     # 特徴量生成パイプライン
│       │   └── store.py        # (Optional) 特徴量ストア連携クライアント
│       ├── models/             # 機械学習モデル関連ロジック
│       │   ├── __init__.py
│       │   ├── base.py         # (Optional) モデルクラスの抽象基底クラス
│       │   ├── types/          # モデル実装 (フレームワーク/アルゴリズムごと)
│       │   │   ├── __init__.py
│       │   │   ├── sklearn_model.py # 学習・評価・推論メソッド、MLflow連携含むクラス
│       │   │   └── xgboost_model.py # 同上
│       │   │   # ... 他のモデルタイプ
│       │   ├── registry.py     # MLflow Model Registry操作ヘルパー (登録, ステージ移行)
│       │   └── utils.py        # モデル関連ユーティリティ
│       ├── pipelines/          # 主要なワークフロー/エントリーポイント
│       │   ├── __init__.py
│       │   ├── train_pipeline.py # データ準備〜学習〜評価の一連の流れ (MLprojectから呼び出し)
│       │   └── batch_predict_pipeline.py # バッチ推論パイプライン
│       ├── databricks/         # Databricks固有コード (必要に応じて)
│       │   ├── __init__.py
│       │   ├── jobs/           # Databricksジョブ定義 (src内の関数を呼び出すラッパー)
│       │   │   └── train_job.py
│       │   └── utils.py        # Databricks API/DBUtils ラッパー等
│       └── orchestration/      # (Optional) ワークフローエンジン連携コード
│           ├── __init__.py
│           └── dags/           # Airflow DAG 定義ファイル
│               └── model_training_dag.py
└── tests/                      # テストコード
    ├── conftest.py             # pytest 共通フィクスチャ (DB接続, MLflow設定, テストデータ生成)
    ├── integration/            # 統合テスト
    │   ├── test_api.py         # APIエンドポイントのテスト (FastAPI TestClient)
    │   └── test_pipelines.py   # 主要パイプラインの結合テスト (例: train_pipeline)
    └── unit/                   # ユニットテスト (srcの構造を反映)
        └── my_project/
            ├── api/
            │   └── test_routers.py
            ├── core/
            │   └── test_config.py
            ├── data_processing/
            │   ├── test_ingest.py
            │   └── test_validate.py
            ├── feature_engineering/
            │   └── test_pipeline.py
            └── models/
                ├── test_sklearn_model.py # MLflow APIのMockを含む
                └── test_registry.py      # MLflow Registry APIのMockを含む
