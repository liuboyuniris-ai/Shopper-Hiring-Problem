-- Q1: "Success" for shopper recruiting funnel achieved if they complete their first batch (see "first_batch_completed_date")

WITH t1 AS (
  SELECT
    exp_group,
    event_type,
    COUNT(DISTINCT applicant_id) AS n_applicants_completed_first_batch
  FROM applicant
  GROUP BY exp_group, event_type
  ORDER BY exp_group, n_applicants_completed_first_batch DESC
),
t2 AS (
  SELECT
    *,
    MAX(n_applicants_completed_first_batch) OVER (PARTITION BY exp_group) AS n_applicants,
    1.0 * n_applicants_completed_first_batch / MAX(n_applicants_completed_first_batch) OVER (PARTITION BY exp_group) AS cvr
  FROM t1
)
SELECT *
FROM t2
WHERE event_type = 'first_batch_completed_date';

-- Q2: Based on the data, what can we conclude from the A/B test?

WITH t1 AS (
  SELECT
    exp_group,
    channel,
    event_type,
    COUNT(DISTINCT applicant_id) AS n_applicants_completed_first_batch
  FROM applicant
  GROUP BY exp_group, channel, event_type
  ORDER BY exp_group, channel, n_applicants_completed_first_batch DESC
),
t2 AS (
  SELECT
    *,
    MAX(n_applicants_completed_first_batch) OVER (PARTITION BY exp_group, channel) AS n_applicants,
    1.0 * n_applicants_completed_first_batch / MAX(n_applicants_completed_first_batch) OVER (PARTITION BY exp_group, channel) AS cvr
  FROM t1
)
SELECT *
FROM t2
WHERE event_type = 'first_batch_completed_date';

-- Q3: Assessing cost-effectiveness and providing recommendations

WITH t1 AS (
  SELECT
    exp_group,
    channel,
    event_type,
    COUNT(DISTINCT applicant_id) AS n_applicants_completed_first_batch
  FROM applicant
  GROUP BY exp_group, channel, event_type
  ORDER BY exp_group, channel, n_applicants_completed_first_batch DESC
),
t2 AS (
  SELECT
    *,
    MAX(n_applicants_completed_first_batch) OVER (PARTITION BY exp_group, channel) AS n_applicants,
    1.0 * n_applicants_completed_first_batch / MAX(n_applicants_completed_first_batch) OVER (PARTITION BY exp_group, channel) AS cvr
  FROM t1
)
SELECT *
FROM t2
WHERE event_type = 'first_batch_completed_date'
ORDER BY 2, 1;
