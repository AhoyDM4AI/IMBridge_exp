-- Select tpcx uc07 Staged Prediction

-- window function
select userID, productID, r, score 
from (select userID, productID, score, rank() OVER (PARTITION BY userID ORDER BY score) as r 
from (select userID, productID, PREDICT tpcx_ai_uc07_staged(userID, productID) score 
from (select userID, productID 
from Product_Rating_sf10
group by userID, productID)))
where r <= 10;

-- temp table
/*
CREATE TABLE temp_uc07(userID INTEGER, productID INTEGER, score DOUBLE);

INSERT INTO temp_uc07 select userID, productID, PREDICT tpcx_ai_uc07_staged(userID, productID) score 
from (select userID, productID 
from Product_Rating_sf10
group by userID, productID);

select userID, productID, r, score 
from (select userID, productID, score, rank() OVER (PARTITION BY userID ORDER BY score) as r 
from temp_uc07) where r <= 10;
*/