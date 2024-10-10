create temp TABLE card_t as
SELECT Time, V1, V2, V3, V4, V5, V6, V7, V8, V9, V10, V11, V12, V13, V14, V15,
 V16, V17, V18, V19, V20, V21, V22, V23, V24, V25, V26, V27, V28, Amount
FROM Credit_Card_extension
WHERE V1 > 1 AND V2 < 0.27 AND V3 > 0.3;

create TEMP TABLE card_t_feats as
select Time, V1/V1_std V1, V2/V2_std V2, V3/V3_std V3,
       V4/V4_std V4, V5/V5_std V5, V6/V6_std V6,
       V7/V7_std V7, V8/V8_std V8, V9/V9_std V9,
       V10/V10_std V10, V11/V11_std V11, V12/V12_std V12,
       V13/V13_std V13, V14/V14_std V14, V15/V15_std V15,
       V16/V16_std V16, V17/V17_std V17, V18/V18_std V18,
       V19/V19_std V19, V20/V20_std V20, V21/V21_std V21,
       V22/V22_std V22, V23/V23_std V23, V24/V24_std V24,
       V25/V25_std V25, V26/V26_std V26, V27/V27_std V27,
       V28/V28_std V28, Amount/Amount_std Amount
from (select Time, V1 - V1_avg V1, V2 - V2_avg V2, V3 - V3_avg V3, V4 - V4_avg V4, V5 - V5_avg V5,
             V6 - V6_avg V6, V7 - V7_avg V7, V8 - V8_avg V8, V9 - V9_avg V9, V10 - V10_avg V10,
             V11 - V11_avg V11, V12 - V12_avg V12, V13 - V13_avg V13, V14 - V14_avg V14,
             V15 - V15_avg V15, V16 - V16_avg V16, V17 - V17_avg V17, V18 - V18_avg V18,
             V19 - V19_avg V19, V20 - V20_avg V20, V21 - V21_avg V21, V22 - V22_avg V22,
             V23 - V23_avg V23, V24 - V24_avg V24, V25 - V25_avg V25, V26 - V26_avg V26,
             V27 - V27_avg V27, V28 - V28_avg V28, Amount - Amount_avg Amount
from card_t cross join card_avgs) t1 cross join card_stds;

drop table if exists card_t_out;

SELECT madlib.forest_predict('card_rf_model',        -- tree model
                             'card_t_feats',             -- new data table
                             'card_t_out',  -- output table
                             'response');           -- show response

explain analyze select * from card_t_out;
