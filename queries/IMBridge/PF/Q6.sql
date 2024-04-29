-- Select Hospital Staged Prediction
SELECT eid, PREDICT hospital_sklearn_lr_staged(hematocrit, neutrophils, sodium, glucose, 
bloodureanitro, creatinine, bmi, pulse, respiration, secondarydiagnosisnonicd9, rcount,
gender, dialysisrenalendstage, asthma, irondef, pneum, substancedependence, psychologicaldisordermajor, 
depress, psychother, fibrosisandother, malnutrition, hemo) AS lengthofstay FROM LengthOfStay_1G;

/* 1.04%
WHERE hematocrit > 12 AND neutrophils > 12 AND bloodureanitro < 13 AND pulse < 60;
*/

/* 10.10%
WHERE hematocrit > 10 AND neutrophils > 10 AND bloodureanitro < 16 AND pulse < 72;
*/

/* 30.13%
WHERE hematocrit > 12 AND neutrophils > 6.5;
*/

/* 60.98%
WHERE hematocrit > 11 and pulse < 85;
*/