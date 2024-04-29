-- Select tpcx uc10 Staged Prediction
select transactionID, PREDICT tpcx_ai_uc10_staged(CAST(amount_norm AS DOUBLE), CAST(business_hour_norm AS DOUBLE)) AS isFraud 
from (select transactionID, amount/transaction_limit amount_norm, HOUR(STR_TO_DATE(time, '%Y-%m-%dT%H:%M'))/23 AS business_hour_norm 
from Financial_Account_sf10 join Financial_Transactions_sf10 on Financial_Account_sf10.fa_customer_sk = Financial_Transactions_sf10.senderID);