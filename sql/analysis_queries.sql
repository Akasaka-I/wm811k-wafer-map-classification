-- Rank models by validation Macro F1.
WITH ranked_models AS(
    SELECT
        model_name,
        model_kind,
        validation_macro_f1,
        validation_balanced_accuracy,
        parameter_count,
        DENSE_RANK() OVER(
            ORDER BY validation_macro_f1 DESC
        ) AS macro_f1_rank
    FROM model_experiments
)

SELECT
    macro_f1_rank,
    model_name,
    model_kind,
    ROUND(validation_macro_f1:: numeric, 4) AS macro_f1,
    ROUND(validation_balanced_accuracy:: numeric, 4)
        AS balanced_accuracy,
    parameter_count
FROM ranked_models
ORDER BY macro_f1_rank, model_name;

-- Find the three weakest classes for each model and split.
WITH ranked_class_performance AS(
    SELECT
        experiment.model_name,
        metric.split_name,
        metric.class_name,
        metric.precision_score,
        metric.recall_score,
        metric.f1_score,
        metric.support_count,
        ROW_NUMBER() OVER(
            PARTITION BY experiment.id, metric.split_name
            ORDER BY metric.f1_score ASC
        ) AS weakness_rank
    FROM model_experiments AS experiment
    JOIN class_metrics AS metric
        ON metric.experiment_id = experiment.id
)
SELECT
    model_name,
    split_name,
    weakness_rank,
    class_name,
    ROUND(precision_score::numeric, 4) AS precision,
    ROUND(recall_score::numeric, 4) AS recall,
    ROUND(f1_score::numeric, 4) AS f1,
    support_count
FROM ranked_class_performance
WHERE weakness_rank <= 3
ORDER BY model_name, split_name, weakness_rank;

-- Summarize prediction errors and high-confidence error rates by class.
WITH error_statistics AS (
    SELECT
        experiment.model_name,
        error.split_name,
        error.true_class,
        COUNT(*) AS total_errors,
        COUNT(*) FILTER (
            WHERE error.confidence >= 0.9
        ) AS high_confidence_errors,
        AVG(error.confidence) AS average_confidence
    FROM model_experiments AS experiment
    JOIN prediction_errors AS error
        ON error.experiment_id = experiment.id
    GROUP BY
        experiment.model_name,
        error.split_name,
        error.true_class
),
ranked_errors AS (
    SELECT
        *,
        DENSE_RANK() OVER (
            PARTITION BY model_name, split_name
            ORDER BY total_errors DESC
        ) AS error_rank
    FROM error_statistics
)
SELECT
    model_name,
    split_name,
    error_rank,
    true_class,
    total_errors,
    high_confidence_errors,
    ROUND(
        100.0 * high_confidence_errors
        / NULLIF(total_errors, 0),
        2
    ) AS high_confidence_percentage,
    ROUND(average_confidence::numeric, 4)
        AS average_confidence
FROM ranked_errors
ORDER BY model_name, split_name, error_rank, true_class;