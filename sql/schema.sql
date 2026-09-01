CREATE TABLE IF NOT EXISTS model_experiments(
    id BIGSERIAL PRIMARY KEY,
    model_name VARCHAR(100) NOT NULL,
    model_kind VARCHAR(30) NOT NULL,
    validation_macro_f1 DOUBLE PRECISION NOT NULL,
    validation_balanced_accuracy DOUBLE PRECISION NOT NULL,
    validation_accuracy DOUBLE PRECISION NOT NULL,
    parameter_count BIGINT,
    history_path TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT uq_model_experiments_name UNIQUE (model_name),
    CONSTRAINT chk_validation_macro_f1
        CHECK (validation_macro_f1 BETWEEN 0 AND 1),
    CONSTRAINT chk_validation_balanced_accuracy
        CHECK (validation_balanced_accuracy BETWEEN 0 AND 1),
    CONSTRAINT chk_validation_accuracy
        CHECK (validation_accuracy BETWEEN 0 AND 1),
    CONSTRAINT chk_parameter_count
        CHECK (parameter_count IS NULL OR parameter_count >= 0)
);

CREATE TABLE IF NOT EXISTS class_metrics (
    id BIGSERIAL PRIMARY KEY,
    experiment_id BIGINT NOT NULL,
    split_name VARCHAR(20) NOT NULL,
    class_name VARCHAR(50) NOT NULL,
    precision_score DOUBLE PRECISION NOT NULL,
    recall_score DOUBLE PRECISION NOT NULL,
    f1_score DOUBLE PRECISION NOT NULL,
    support_count INTEGER NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT fk_class_metrics_experiment
        FOREIGN KEY (experiment_id)
        REFERENCES model_experiments (id)
        ON DELETE CASCADE,

    CONSTRAINT uq_class_metrics_experiment_split_class
        UNIQUE (experiment_id, split_name, class_name),

    CONSTRAINT chk_class_metrics_split
        CHECK (split_name IN ('validation', 'test')),

    CONSTRAINT chk_class_metrics_precision
        CHECK (precision_score BETWEEN 0 AND 1),

    CONSTRAINT chk_class_metrics_recall
        CHECK (recall_score BETWEEN 0 AND 1),

    CONSTRAINT chk_class_metrics_f1
        CHECK (f1_score BETWEEN 0 AND 1),

    CONSTRAINT chk_class_metrics_support
        CHECK (support_count >= 0)
);

CREATE TABLE IF NOT EXISTS prediction_errors (
    id BIGSERIAL PRIMARY KEY,
    experiment_id BIGINT NOT NULL,
    split_name VARCHAR(20) NOT NULL,
    dataset_position INTEGER NOT NULL,
    global_index INTEGER NOT NULL,
    lot_name VARCHAR(100) NOT NULL,
    true_class VARCHAR(50) NOT NULL,
    predicted_class VARCHAR(50) NOT NULL,
    confidence DOUBLE PRECISION NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT fk_prediction_errors_experiment
        FOREIGN KEY (experiment_id)
        REFERENCES model_experiments (id)
        ON DELETE CASCADE,

    CONSTRAINT uq_prediction_errors_experiment_sample
        UNIQUE (experiment_id, split_name, global_index),

    CONSTRAINT chk_prediction_errors_split
        CHECK (split_name IN ('validation', 'test')),

    CONSTRAINT chk_prediction_errors_position
        CHECK (dataset_position >= 0),

    CONSTRAINT chk_prediction_errors_global_index
        CHECK (global_index >= 0),

    CONSTRAINT chk_prediction_errors_confidence
        CHECK (confidence BETWEEN 0 AND 1),

    CONSTRAINT chk_prediction_errors_is_error
        CHECK (true_class <> predicted_class)
);

CREATE INDEX IF NOT EXISTS idx_prediction_errors_lot
    ON prediction_errors (lot_name);

CREATE INDEX IF NOT EXISTS idx_prediction_errors_class_pair
    ON prediction_errors (true_class, predicted_class);