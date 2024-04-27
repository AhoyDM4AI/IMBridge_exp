-- Select Flights Staged Prediction
SELECT Flights_S_routes_1G.airlineid, Flights_S_routes_1G.sairportid, Flights_S_routes_1G.dairportid, 
PREDICT flights_sklearn_rf_staged(slatitude, slongitude, dlatitude, dlongitude, name1, name2, name4, acountry, 
active, scity, scountry, stimezone, sdst, dcity, dcountry, dtimezone, ddst) AS codeshare 
FROM Flights_S_routes_1G JOIN Flights_R1_airlines_1G ON Flights_S_routes_1G.airlineid = Flights_R1_airlines_1G.airlineid 
JOIN Flights_R2_sairports_1G ON Flights_S_routes_1G.sairportid = Flights_R2_sairports_1G.sairportid 
JOIN Flights_R3_dairports_1G ON Flights_S_routes_1G.dairportid = Flights_R3_dairports_1G.dairportid;

/* 0.88%
WHERE name4 = 'f' and sdst = 'U' and ddst = 'E' and name1 > 1;
*/

/* 9.97%
WHERE name2 = 't' and name4 = 't' and name1 = 3;
*/

/* 29.78%
WHERE name2 = 't' and (sdst = 'A' or sdst = 'N');
*/

/* 60.34%
WHERE name4 = 'f' and name1 < 4;
*/