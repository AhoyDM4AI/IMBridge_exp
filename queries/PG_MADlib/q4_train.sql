CREATE TABLE hospital_stds as
select STDDEV(hematocrit) hematocrit_std, STDDEV(neutrophils) neutrophils_std, STDDEV(sodium) sodium_std,
       STDDEV(glucose) glucose_std, STDDEV(bloodureanitro) bloodureanitro_std, STDDEV(creatinine) creatinine_std,
       STDDEV(bmi) bmi_std, STDDEV(pulse) pulse_std, STDDEV(respiration) respiration_std,
       STDDEV(secondarydiagnosisnonicd9) secondarydiagnosisnonicd9_std
from lengthofstay;

CREATE TABLE hospital_avgs as
select AVG(hematocrit) hematocrit_avg, AVG(neutrophils) neutrophils_avg,
       AVG(sodium) sodium_avg, AVG(glucose) glucose_avg, AVG(bloodureanitro) bloodureanitro_avg,
       AVG(creatinine) creatinine_avg, AVG(bmi) bmi_avg, AVG(pulse) pulse_avg,
       AVG(respiration) respiration_avg, AVG(secondarydiagnosisnonicd9) secondarydiagnosisnonicd9_avg
from lengthofstay;

create TEMP TABLE hospital_feats as
select eid, hematocrit/hematocrit_std hematocrit, neutrophils/neutrophils_std neutrophils,
       sodium/sodium_std sodium, glucose/glucose_std glucose, bloodureanitro/bloodureanitro_std bloodureanitro,
       creatinine/creatinine_std creatinine, bmi/bmi_std bmi, pulse/pulse_std pulse,
       respiration/respiration_std respiration, secondarydiagnosisnonicd9/secondarydiagnosisnonicd9_std secondarydiagnosisnonicd9,
        rcount, gender, dialysisrenalendstage, asthma,
        irondef, pneum, substancedependence,
        psychologicaldisordermajor, depress, psychother,
        fibrosisandother, malnutrition, hemo, lengthofstay
from (select eid, hematocrit - hematocrit_avg hematocrit, neutrophils - neutrophils_avg neutrophils,
             sodium - sodium_avg sodium, glucose - glucose_avg glucose,
             bloodureanitro - bloodureanitro_avg bloodureanitro,
             creatinine - creatinine_avg creatinine, bmi - bmi_avg bmi,
             pulse - pulse_avg pulse, respiration - respiration_avg respiration,
             secondarydiagnosisnonicd9 - secondarydiagnosisnonicd9_avg secondarydiagnosisnonicd9,
            rcount, gender, dialysisrenalendstage, asthma,
            irondef, pneum, substancedependence,
            psychologicaldisordermajor, depress, psychother,
            fibrosisandother, malnutrition, hemo, lengthofstay
from lengthofstay cross join hospital_avgs) t1 cross join hospital_stds;

select (hematocrit-(select AVG(hematocrit) from lengthofstay))/(select STDDEV(hematocrit) from lengthofstay),
       (neutrophils-(select AVG(neutrophils) from lengthofstay))/(select STDDEV(neutrophils) from lengthofstay),
       (sodium-(select AVG(sodium) from lengthofstay))/(select STDDEV(sodium) from lengthofstay),
       (glucose-(select AVG(glucose) from lengthofstay))/(select STDDEV(glucose) from lengthofstay),
       (bloodureanitro-(select AVG(bloodureanitro) from lengthofstay))/(select STDDEV(bloodureanitro) from lengthofstay),
       (creatinine-(select AVG(creatinine) from lengthofstay))/(select STDDEV(creatinine) from lengthofstay),
       (bmi-(select AVG(bmi) from lengthofstay))/(select STDDEV(bmi) from lengthofstay),
       (pulse-(select AVG(pulse) from lengthofstay))/(select STDDEV(pulse) from lengthofstay),
       (respiration-(select AVG(respiration) from lengthofstay))/(select STDDEV(respiration) from lengthofstay),
       (secondarydiagnosisnonicd9-(select AVG(secondarydiagnosisnonicd9) from lengthofstay))/(select STDDEV(secondarydiagnosisnonicd9) from lengthofstay)
from lengthofstay;



SELECT eid, hematocrit, neutrophils, sodium, glucose, bloodureanitro, creatinine, bmi, pulse,
 respiration, secondarydiagnosisnonicd9, rcount, gender, dialysisrenalendstage, asthma,
irondef, pneum, substancedependence,
psychologicaldisordermajor, depress, psychother,
fibrosisandother, malnutrition, hemo from lengthofstay;

SELECT madlib.encode_categorical_variables ('hospital_feats', 'hospital_feats_out',
'rcount, gender, dialysisrenalendstage, asthma,
irondef, pneum, substancedependence,
psychologicaldisordermajor, depress, psychother,
fibrosisandother, malnutrition, hemo'
,NULL,'eid,hematocrit, neutrophils, sodium, glucose, bloodureanitro, creatinine, bmi, pulse,
 respiration, secondarydiagnosisnonicd9, lengthofstay');


SELECT madlib.mlp_regression( 
'hospital_feats_out',
'hospital_mlp',
'ARRAY[1, "hematocrit","neutrophils","sodium","glucose","bloodureanitro","creatinine","bmi","pulse","respiration","secondarydiagnosisnonicd9","rcount_0","rcount_1","rcount_2","rcount_3","rcount_4","rcount_5+","gender_F","gender_M","dialysisrenalendstage_false","dialysisrenalendstage_true","asthma_false","asthma_true","irondef_false","irondef_true","pneum_false","pneum_true","substancedependence_false","substancedependence_true","psychologicaldisordermajor_false","psychologicaldisordermajor_true","depress_false","depress_true","psychother_false","psychother_true","fibrosisandother_false","fibrosisandother_true","malnutrition_false","malnutrition_true","hemo_false","hemo_true"]',
'lengthofstay',
ARRAY[5],         -- Number of units per layer
'learning_rate_init=0.003,
n_iterations=500,
tolerance=0',     -- Optimizer params
'tanh',           -- Activation function
NULL,             -- Default weight (1)
FALSE,
FALSE
);