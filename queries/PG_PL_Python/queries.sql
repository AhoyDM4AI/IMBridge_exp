SET statement_timeout = 10800000;
--q1
SELECT 1;
EXPLAIN ANALYZE SELECT Expedia_R1_hotels.prop_id, Expedia_R2_searches.srch_id, udf1(prop_location_score1, prop_location_score2, prop_log_historical_price, price_usd,
                           orig_destination_distance, prop_review_score, avg_bookings_usd, stdev_bookings_usd,
                           position, prop_country_id, prop_starrating, cast(prop_brand_bool as INTEGER), count_clicks, count_bookings,
                           year, month, weekofyear, time, site_id, visitor_location_country_id, srch_destination_id,
                           srch_length_of_stay, srch_booking_window, srch_adults_count, srch_children_count,
                           srch_room_count, cast(srch_saturday_night_bool as INTEGER), cast(random_bool as INTEGER))
FROM Expedia_S_listings_extension JOIN Expedia_R1_hotels ON Expedia_S_listings_extension.prop_id = Expedia_R1_hotels.prop_id
JOIN Expedia_R2_searches ON Expedia_S_listings_extension.srch_id = Expedia_R2_searches.srch_id
WHERE prop_location_score1 > 1 and prop_location_score2 > 0.1
and prop_log_historical_price > 4 and count_bookings > 5
and srch_booking_window > 10 and srch_length_of_stay > 1;

--q2
SELECT 2;
EXPLAIN ANALYZE SELECT Flights_S_routes_extension.airlineid, Flights_S_routes_extension.sairportid, Flights_S_routes_extension.dairportid,
 udf3(slatitude, slongitude, dlatitude, dlongitude, name1, name2, name4, acountry, active,
 scity, scountry, stimezone, sdst, dcity, dcountry, dtimezone, ddst) AS codeshare
 FROM Flights_S_routes_extension JOIN Flights_R1_airlines ON Flights_S_routes_extension.airlineid = Flights_R1_airlines.airlineid
 JOIN Flights_R2_sairports ON Flights_S_routes_extension.sairportid = Flights_R2_sairports.sairportid JOIN Flights_R3_dairports
 ON Flights_S_routes_extension.dairportid = Flights_R3_dairports.dairportid
WHERE name2 = 't' and name4 = 't' and name1 > 2.8;

--q3
SELECT 3;
EXPLAIN ANALYZE SELECT Time, Amount, udf6(V1, V2, V3, V4, V5, V6, V7, V8, V9, V10, V11, V12, V13, V14, V15,
 V16, V17, V18, V19, V20, V21, V22, V23, V24, V25, V26, V27, V28, Amount) AS Class FROM Credit_Card_extension
 WHERE V1 > 1 AND V2 < 0.27 AND V3 > 0.3;


-- q4 tpch q5
select 4;
explain analyze select
 n_name,
 sum(l_extendedprice * (1 - l_discount)) as revenue
from
 customer,
 orders,
 lineitem,
 supplier,
 nation,
 region
where
 c_custkey = o_custkey
 and l_orderkey = o_orderkey
 and l_suppkey = s_suppkey
 and c_nationkey = s_nationkey
 and s_nationkey = n_nationkey
 and n_regionkey = r_regionkey
 and r_name = 'ASIA'
 and o_orderdate >= date '1994-01-01'
 and o_orderdate < date '1994-01-01' + interval '1' year
 and tpch5(cast(c_acctbal as INTEGER), cast(o_totalprice as INTEGER), cast(l_quantity as INTEGER),
  cast(l_extendedprice as INTEGER), cast(l_discount as INTEGER), cast(l_tax as INTEGER), cast(s_acctbal as INTEGER),
  cast(o_orderstatus as VARCHAR), cast(l_returnflag as VARCHAR), cast(l_linestatus as VARCHAR),
   cast(l_shipinstruct as VARCHAR), cast(l_shipmode as VARCHAR), cast(n_nationkey as INTEGER), cast(n_regionkey as INTEGER)) = '1-URGENT'
group by
 n_name
order by
revenue desc;


 -- q5 tpch q10
select 5;
explain analyze SELECT c_custkey,
               c_name,
               sum(l_extendedprice * (1 - l_discount)) as revenue,
               c_acctbal,
               n_name,
               c_address,
               c_phone,
               c_comment
       from customer, orders, lineitem, nation
       where c_custkey = o_custkey and
             l_orderkey = o_orderkey and
             o_orderdate >= DATE'1993-10-01' and
             o_orderdate < DATE'1993-10-01' + interval '3' month and
             c_nationkey = n_nationkey and 
             tpch10(cast(c_acctbal as INTEGER), cast(o_totalprice as INTEGER), cast(l_quantity as INTEGER),
              cast(l_extendedprice as INTEGER), cast(l_discount as INTEGER), cast(l_tax as INTEGER), 
                   cast(o_orderstatus as VARCHAR), cast(o_orderpriority as VARCHAR), cast(l_linestatus as VARCHAR),
                    cast(l_shipinstruct as VARCHAR), cast(l_shipmode as VARCHAR), cast(n_nationkey as INTEGER), cast(n_regionkey as INTEGER)) = 'R'
       group by c_custkey, c_name, c_acctbal, c_phone, n_name, c_address, c_comment
order by 
revenue desc;

--q6
SELECT 6;
EXPLAIN ANALYZE select store, department, udf8(cast(store as text), cast(department as text))
from (select store, department
from (Order_o Join Lineitem on Order_o.o_order_id = Lineitem.li_order_id) data
Join Product on data.li_product_id=Product.p_product_id
group by store,department) t1;

--q7
SELECT 7;
EXPLAIN ANALYZE select udf9(cast(txt as text)) from
(select DISTINCT text txt from Review) t1;

--q11
SELECT 8;
EXPLAIN ANALYZE select userID, productID, r, score
from (select userID, productID, score, rank() OVER (PARTITION BY userID ORDER BY score) as r
from (select userID, productID, udf11(userID, productID) score
from (select userID, productID
from Product_Rating
group by userID, productID) t3
) t2
) t1
where r <=10;

