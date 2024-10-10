CREATE Temp TABLE hospital_t as
SELECT eid, hematocrit, neutrophils, sodium, glucose, bloodureanitro, creatinine, bmi, pulse,
 respiration, secondarydiagnosisnonicd9, rcount, gender, dialysisrenalendstage, asthma,
irondef, pneum, substancedependence,
psychologicaldisordermajor, depress, psychother,
fibrosisandother, malnutrition, hemo
from LengthOfStay_extension
WHERE hematocrit > 10 AND neutrophils > 10 AND bloodureanitro < 20 AND pulse < 70;

CREATE Temp Table hospital_t_feats as
select eid, hematocrit/hematocrit_std hematocrit, neutrophils/neutrophils_std neutrophils,
       sodium/sodium_std sodium, glucose/glucose_std glucose, bloodureanitro/bloodureanitro_std bloodureanitro,
       creatinine/creatinine_std creatinine, bmi/bmi_std bmi, pulse/pulse_std pulse,
       respiration/respiration_std respiration, secondarydiagnosisnonicd9/secondarydiagnosisnonicd9_std secondarydiagnosisnonicd9,
        rcount, gender, dialysisrenalendstage, asthma,
        irondef, pneum, substancedependence,
        psychologicaldisordermajor, depress, psychother,
        fibrosisandother, malnutrition, hemo
from (select eid, hematocrit - hematocrit_avg hematocrit, neutrophils - neutrophils_avg neutrophils,
             sodium - sodium_avg sodium, glucose - glucose_avg glucose,
             bloodureanitro - bloodureanitro_avg bloodureanitro,
             creatinine - creatinine_avg creatinine, bmi - bmi_avg bmi,
             pulse - pulse_avg pulse, respiration - respiration_avg respiration,
             secondarydiagnosisnonicd9 - secondarydiagnosisnonicd9_avg secondarydiagnosisnonicd9,
            rcount, gender, dialysisrenalendstage, asthma,
            irondef, pneum, substancedependence,
            psychologicaldisordermajor, depress, psychother,
            fibrosisandother, malnutrition, hemo
from hospital_t cross join hospital_avgs) t1 cross join hospital_stds;

drop table if exists hospital_feats_t_out;
SELECT madlib.encode_categorical_variables ('hospital_t_feats', 'hospital_feats_t_out',
'rcount, gender, dialysisrenalendstage, asthma,
irondef, pneum, substancedependence,
psychologicaldisordermajor, depress, psychother,
fibrosisandother, malnutrition, hemo'
,NULL,'eid,hematocrit, neutrophils, sodium, glucose, bloodureanitro, creatinine, bmi, pulse,
 respiration, secondarydiagnosisnonicd9');

-- Add the id column for prediction function
ALTER TABLE hospital_feats_t_out ADD COLUMN id SERIAL;
-- Predict probabilities for all categories using the original data
drop table if exists hospital_t_out;
SELECT madlib.mlp_predict('hospital_mlp','hospital_feats_t_out', 'id', 'hospital_t_out', 'response');
explain analyze select * from hospital_t_out;
