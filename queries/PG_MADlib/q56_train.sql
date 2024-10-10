CREATE TABLE card_stds as
select
STDDEV(V1) V1_std, STDDEV(V2) V2_std, STDDEV(V3) V3_std, STDDEV(V4) V4_std, STDDEV(V5) V5_std, STDDEV(V6) V6_std, STDDEV(V7) V7_std, STDDEV(V8) V8_std, STDDEV(V9) V9_std, STDDEV(V10) V10_std, STDDEV(V11) V11_std, STDDEV(V12) V12_std, STDDEV(V13) V13_std, STDDEV(V14) V14_std, STDDEV(V15) V15_std, STDDEV(V16) V16_std, STDDEV(V17) V17_std, STDDEV(V18) V18_std, STDDEV(V19) V19_std, STDDEV(V20) V20_std, STDDEV(V21) V21_std,
STDDEV(V22) V22_std, STDDEV(V23) V23_std, STDDEV(V24) V24_std, STDDEV(V25) V25_std, STDDEV(V26) V26_std, STDDEV(V27) V27_std, STDDEV(V28) V28_std, STDDEV(Amount) Amount_std
from credit_card;

CREATE TABLE card_avgs as
select
AVG(V1) V1_avg, AVG(V2) V2_avg, AVG(V3) V3_avg, AVG(V4) V4_avg, AVG(V5) V5_avg, AVG(V6) V6_avg, AVG(V7) V7_avg, AVG(V8) V8_avg, AVG(V9) V9_avg, AVG(V10) V10_avg, AVG(V11) V11_avg, AVG(V12) V12_avg, AVG(V13) V13_avg, AVG(V14) V14_avg, AVG(V15) V15_avg, AVG(V16) V16_avg, AVG(V17) V17_avg, AVG(V18) V18_avg, AVG(V19) V19_avg, AVG(V20) V20_avg,
AVG(V21) V21_avg, AVG(V22) V22_avg, AVG(V23) V23_avg, AVG(V24) V24_avg, AVG(V25) V25_avg, AVG(V26) V26_avg, AVG(V27) V27_avg, AVG(V28) V28_avg, AVG(Amount) Amount_avg
from credit_card;

create TEMP TABLE card_feats as
select Time, V1/V1_std V1, V2/V2_std V2, V3/V3_std V3,
       V4/V4_std V4, V5/V5_std V5, V6/V6_std V6,
       V7/V7_std V7, V8/V8_std V8, V9/V9_std V9,
       V10/V10_std V10, V11/V11_std V11, V12/V12_std V12,
       V13/V13_std V13, V14/V14_std V14, V15/V15_std V15,
       V16/V16_std V16, V17/V17_std V17, V18/V18_std V18,
       V19/V19_std V19, V20/V20_std V20, V21/V21_std V21,
       V22/V22_std V22, V23/V23_std V23, V24/V24_std V24,
       V25/V25_std V25, V26/V26_std V26, V27/V27_std V27,
       V28/V28_std V28, Amount/Amount_std Amount, class
from (select Time, V1 - V1_avg V1, V2 - V2_avg V2, V3 - V3_avg V3, V4 - V4_avg V4, V5 - V5_avg V5,
             V6 - V6_avg V6, V7 - V7_avg V7, V8 - V8_avg V8, V9 - V9_avg V9, V10 - V10_avg V10,
             V11 - V11_avg V11, V12 - V12_avg V12, V13 - V13_avg V13, V14 - V14_avg V14,
             V15 - V15_avg V15, V16 - V16_avg V16, V17 - V17_avg V17, V18 - V18_avg V18,
             V19 - V19_avg V19, V20 - V20_avg V20, V21 - V21_avg V21, V22 - V22_avg V22,
             V23 - V23_avg V23, V24 - V24_avg V24, V25 - V25_avg V25, V26 - V26_avg V26,
             V27 - V27_avg V27, V28 - V28_avg V28, Amount - Amount_avg Amount, class
from credit_card cross join card_avgs) t1 cross join card_stds;


SELECT madlib.forest_train('card_feats',         -- source table
'card_rf_model',    -- output model table
'Time',              -- id column
'class',           -- response
'V1, V2, V3, V4, V5, V6, V7, V8, V9, V10, V11, V12, V13, V14, V15,  V16, V17, V18, V19, V20, V21, V22, V23, V24, V25, V26, V27, V28, Amount',   -- features
NULL,              -- exclude columns
NULL,              -- grouping columns
20::integer,       -- number of trees
2::integer,        -- number of random features
TRUE::boolean,     -- variable importance
1::integer,        -- num_permutations
8::integer,        -- max depth
3::integer,        -- min split
1::integer,        -- min bucket
10::integer,        -- number of splits per continuous variable
NULL,
true
);