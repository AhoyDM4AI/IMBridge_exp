-- Select Credit Card Staged Prediction
SELECT Time, Amount, PREDICT creditcard_lightgbm_gb_staged(V1, V2, V3, V4, V5, V6, V7, V8, V9, V10, V11, V12, 
V13, V14, V15, V16, V17, V18, V19, V20, V21, V22, V23, V24, V25, V26, V27, V28, Amount) AS Class FROM Credit_Card_1G;

/* 1.04%
WHERE V1 > 1.15 AND V2 < 0.2 AND V3 > 1;
*/

/* 9.99%
WHERE V1 > 1 AND V2 < 0.27 AND V3 > 0.3;
*/

/* 30.07%
WHERE V1 > 1.15 AND V2 < 0.23;
*/

/* 60.49%
WHERE V1 < 1.05;
*/
