-- Select tpcx uc04 Staged Prediction
select ID, PREDICT tpcx_ai_uc04_staged(text) from Review_sf10;