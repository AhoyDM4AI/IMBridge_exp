-- Select tpcx uc08 Staged Prediction
-- one-hot encoder in sql
select o_order_id, PREDICT tpcx_ai_uc08_staged(cast(scan_count as SIGNED), cast(scan_count_abs as SIGNED), Monday, Tuesday, Wednesday, Thursday, Friday, Saturday, Sunday, cast(dep0 as SIGNED), cast(dep1 as SIGNED), cast(dep2 as SIGNED), cast(dep3 as SIGNED), cast(dep4 as SIGNED), cast(dep5 as SIGNED), cast(dep6 as SIGNED), cast(dep7 as SIGNED), cast(dep8 as SIGNED), cast(dep9 as SIGNED), cast(dep10 as SIGNED), cast(dep11 as SIGNED), cast(dep12 as SIGNED), cast(dep13 as SIGNED), cast(dep14 as SIGNED), cast(dep15 as SIGNED), cast(dep16 as SIGNED), cast(dep17 as SIGNED), cast(dep18 as SIGNED), cast(dep19 as SIGNED), cast(dep20 as SIGNED), cast(dep21 as SIGNED), cast(dep22 as SIGNED), cast(dep23 as SIGNED), cast(dep24 as SIGNED), cast(dep25 as SIGNED), cast(dep26 as SIGNED), cast(dep27 as SIGNED), cast(dep28 as SIGNED), cast(dep29 as SIGNED), cast(dep30 as SIGNED), cast(dep31 as SIGNED), cast(dep32 as SIGNED), cast(dep33 as SIGNED), cast(dep34 as SIGNED), cast(dep35 as SIGNED), cast(dep36 as SIGNED), cast(dep37 as SIGNED), cast(dep38 as SIGNED), cast(dep39 as SIGNED), cast(dep40 as SIGNED), cast(dep41 as SIGNED), cast(dep42 as SIGNED), cast(dep43 as SIGNED), cast(dep44 as SIGNED), cast(dep45 as SIGNED), cast(dep46 as SIGNED), cast(dep47 as SIGNED), cast(dep48 as SIGNED), cast(dep49 as SIGNED), cast(dep50 as SIGNED), cast(dep51 as SIGNED), cast(dep52 as SIGNED), cast(dep53 as SIGNED), cast(dep54 as SIGNED), cast(dep55 as SIGNED), cast(dep56 as SIGNED), cast(dep57 as SIGNED), cast(dep58 as SIGNED), cast(dep59 as SIGNED), cast(dep60 as SIGNED), cast(dep61 as SIGNED), cast(dep62 as SIGNED), cast(dep63 as SIGNED), cast(dep64 as SIGNED), cast(dep65 as SIGNED), cast(dep66 as SIGNED), cast(dep67 as SIGNED)) AS trip_type from ( 
select t1.o_order_id, scan_count, scan_count_abs, Monday, Tuesday, Wednesday, Thursday, Friday, Saturday, Sunday, dep0, dep1, dep2, dep3, dep4, dep5, dep6, dep7, dep8, dep9, dep10, dep11, dep12, dep13, dep14, dep15, dep16, dep17, dep18, dep19, dep20, dep21, dep22, dep23, dep24, dep25, dep26, dep27, dep28, dep29, dep30, dep31, dep32, dep33, dep34, dep35, dep36, dep37, dep38, dep39, dep40, dep41, dep42, dep43, dep44, dep45, dep46, dep47, dep48, dep49, dep50, dep51, dep52, dep53, dep54, dep55, dep56, dep57, dep58, dep59, dep60, dep61, dep62, dep63, dep64, dep65, dep66, dep67 from (
select o_order_id, sum(scan_count) scan_count, sum(abs(scan_count)) scan_count_abs 
from (select Order_o_sf10.o_order_id, quantity scan_count 
from Order_o_sf10 join Lineitem_sf10 on Order_o_sf10.o_order_id = Lineitem_sf10.li_order_id
join Product_sf10 on Lineitem_sf10.li_product_id = Product_sf10.p_product_id
) group by o_order_id
) t1 join (
select o_order_id, cast(count(Monday)>0 as SIGNED) Monday,  cast(count(Tuesday) >0 as SIGNED) Tuesday, cast(count(Wednesday)>0 as SIGNED) Wednesday,
 cast(count(Thursday)>0 as SIGNED) Thursday, cast(count(Friday)>0 as SIGNED) Friday, cast(count(Saturday)>0 as SIGNED) Saturday, cast(count(Sunday)>0 as SIGNED) Sunday 
from (select o_order_id,
    case when weekday='Monday' then scan_count end Monday,
    case when weekday='Tuesday' then scan_count end Tuesday,
    case when weekday='Wednesday' then scan_count end Wednesday,
    case when weekday='Thursday' then scan_count end Thursday,
    case when weekday='Friday' then scan_count end Friday,
    case when weekday='Saturday' then scan_count end Saturday,
    case when weekday='Sunday' then scan_count end Sunday 
from (select Order_o_sf10.o_order_id, dayname(cast(date as Date)) weekday, quantity scan_count 
from Order_o_sf10 join Lineitem_sf10 on Order_o_sf10.o_order_id = Lineitem_sf10.li_order_id
join Product_sf10 on Lineitem_sf10.li_product_id = Product_sf10.p_product_id))
group by o_order_id
) t2 on t1.o_order_id = t2.o_order_id
join (
select o_order_id, COALESCE(sum(dep0),0) dep0, COALESCE(sum(dep1),0) dep1, COALESCE(sum(dep2),0) dep2, COALESCE(sum(dep3),0) dep3,
 COALESCE(sum(dep4),0) dep4, COALESCE(sum(dep5),0) dep5, COALESCE(sum(dep6),0) dep6, COALESCE(sum(dep7),0) dep7, COALESCE(sum(dep8),0) dep8,
  COALESCE(sum(dep9),0) dep9, COALESCE(sum(dep10),0) dep10, COALESCE(sum(dep11),0) dep11, COALESCE(sum(dep12),0) dep12, COALESCE(sum(dep13),0) dep13,
   COALESCE(sum(dep14),0) dep14, COALESCE(sum(dep15),0) dep15, COALESCE(sum(dep16),0) dep16, COALESCE(sum(dep17),0) dep17, COALESCE(sum(dep18),0) dep18,
    COALESCE(sum(dep19),0) dep19, COALESCE(sum(dep20),0) dep20, COALESCE(sum(dep21),0) dep21, COALESCE(sum(dep22),0) dep22, COALESCE(sum(dep23),0) dep23,
     COALESCE(sum(dep24),0) dep24, COALESCE(sum(dep25),0) dep25, COALESCE(sum(dep26),0) dep26, COALESCE(sum(dep27),0) dep27, COALESCE(sum(dep28),0) dep28,
      COALESCE(sum(dep29),0) dep29, COALESCE(sum(dep30),0) dep30, COALESCE(sum(dep31),0) dep31, COALESCE(sum(dep32),0) dep32, COALESCE(sum(dep33),0) dep33,
       COALESCE(sum(dep34),0) dep34, COALESCE(sum(dep35),0) dep35, COALESCE(sum(dep36),0) dep36, COALESCE(sum(dep37),0) dep37, COALESCE(sum(dep38),0) dep38,
        COALESCE(sum(dep39),0) dep39, COALESCE(sum(dep40),0) dep40, COALESCE(sum(dep41),0) dep41, COALESCE(sum(dep42),0) dep42, COALESCE(sum(dep43),0) dep43,
         COALESCE(sum(dep44),0) dep44, COALESCE(sum(dep45),0) dep45, COALESCE(sum(dep46),0) dep46, COALESCE(sum(dep47),0) dep47, COALESCE(sum(dep48),0) dep48,
          COALESCE(sum(dep49),0) dep49, COALESCE(sum(dep50),0) dep50, COALESCE(sum(dep51),0) dep51, COALESCE(sum(dep52),0) dep52, COALESCE(sum(dep53),0) dep53,
           COALESCE(sum(dep54),0) dep54, COALESCE(sum(dep55),0) dep55, COALESCE(sum(dep56),0) dep56, COALESCE(sum(dep57),0) dep57, COALESCE(sum(dep58),0) dep58,
            COALESCE(sum(dep59),0) dep59, COALESCE(sum(dep60),0) dep60, COALESCE(sum(dep61),0) dep61, COALESCE(sum(dep62),0) dep62, COALESCE(sum(dep63),0) dep63,
             COALESCE(sum(dep64),0) dep64, COALESCE(sum(dep65),0) dep65, COALESCE(sum(dep66),0) dep66, COALESCE(sum(dep67),0) dep67 
             from (select o_order_id, case when department='FINANCIAL SERVICES' then scan_count end dep0, 
case when department='SHOES' then scan_count end dep1, 
case when department='PERSONAL CARE' then scan_count end dep2, 
case when department='PAINT AND ACCESSORIES' then scan_count end dep3, 
case when department='DSD GROCERY' then scan_count end dep4, 
case when department='MEAT - FRESH & FROZEN' then scan_count end dep5, 
case when department='DAIRY' then scan_count end dep6, 
case when department='PETS AND SUPPLIES' then scan_count end dep7, 
case when department='HOUSEHOLD CHEMICALS/SUPP' then scan_count end dep8, 
case when department='IMPULSE MERCHANDISE' then scan_count end dep9, 
case when department='PRODUCE' then scan_count end dep10, 
case when department='CANDY, TOBACCO, COOKIES' then scan_count end dep11, 
case when department='GROCERY DRY GOODS' then scan_count end dep12, 
case when department='BOYS WEAR' then scan_count end dep13, 
case when department='FABRICS AND CRAFTS' then scan_count end dep14, 
case when department='JEWELRY AND SUNGLASSES' then scan_count end dep15, 
case when department='MENS WEAR' then scan_count end dep16, 
case when department='ACCESSORIES' then scan_count end dep17, 
case when department='HOME MANAGEMENT' then scan_count end dep18, 
case when department='FROZEN FOODS' then scan_count end dep19, 
case when department='SERVICE DELI' then scan_count end dep20, 
case when department='INFANT CONSUMABLE HARDLINES' then scan_count end dep21, 
case when department='PRE PACKED DELI' then scan_count end dep22, 
case when department='COOK AND DINE' then scan_count end dep23, 
case when department='PHARMACY OTC' then scan_count end dep24, 
case when department='LADIESWEAR' then scan_count end dep25, 
case when department='COMM BREAD' then scan_count end dep26, 
case when department='BAKERY' then scan_count end dep27, 
case when department='HOUSEHOLD PAPER GOODS' then scan_count end dep28, 
case when department='CELEBRATION' then scan_count end dep29, 
case when department='HARDWARE' then scan_count end dep30, 
case when department='BEAUTY' then scan_count end dep31, 
case when department='AUTOMOTIVE' then scan_count end dep32, 
case when department='BOOKS AND MAGAZINES' then scan_count end dep33, 
case when department='SEAFOOD' then scan_count end dep34, 
case when department='OFFICE SUPPLIES' then scan_count end dep35, 
case when department='LAWN AND GARDEN' then scan_count end dep36, 
case when department='SHEER HOSIERY' then scan_count end dep37, 
case when department='WIRELESS' then scan_count end dep38, 
case when department='BEDDING' then scan_count end dep39, 
case when department='BATH AND SHOWER' then scan_count end dep40, 
case when department='HORTICULTURE AND ACCESS' then scan_count end dep41, 
case when department='HOME DECOR' then scan_count end dep42, 
case when department='TOYS' then scan_count end dep43, 
case when department='INFANT APPAREL' then scan_count end dep44, 
case when department='LADIES SOCKS' then scan_count end dep45, 
case when department='PLUS AND MATERNITY' then scan_count end dep46, 
case when department='ELECTRONICS' then scan_count end dep47, 
case when department='GIRLS WEAR, 4-6X  AND 7-14' then scan_count end dep48, 
case when department='BRAS & SHAPEWEAR' then scan_count end dep49, 
case when department='LIQUOR,WINE,BEER' then scan_count end dep50, 
case when department='SLEEPWEAR/FOUNDATIONS' then scan_count end dep51, 
case when department='CAMERAS AND SUPPLIES' then scan_count end dep52, 
case when department='SPORTING GOODS' then scan_count end dep53, 
case when department='PLAYERS AND ELECTRONICS' then scan_count end dep54, 
case when department='PHARMACY RX' then scan_count end dep55, 
case when department='MENSWEAR' then scan_count end dep56, 
case when department='OPTICAL - FRAMES' then scan_count end dep57, 
case when department='SWIMWEAR/OUTERWEAR' then scan_count end dep58, 
case when department='OTHER DEPARTMENTS' then scan_count end dep59, 
case when department='MEDIA AND GAMING' then scan_count end dep60, 
case when department='FURNITURE' then scan_count end dep61, 
case when department='OPTICAL - LENSES' then scan_count end dep62, 
case when department='SEASONAL' then scan_count end dep63, 
case when department='LARGE HOUSEHOLD GOODS' then scan_count end dep64, 
case when department='1-HR PHOTO' then scan_count end dep65, 
case when department='CONCEPT STORES' then scan_count end dep66, 
case when department='HEALTH AND BEAUTY AIDS' then scan_count end dep67 from (select Order_o.o_order_id, department, quantity scan_count 
from Order_o_sf10 join Lineitem_sf10 on Order_o_sf10.o_order_id = Lineitem_sf10.li_order_id
join Product_sf10 on Lineitem_sf10.li_product_id = Product_sf10.p_product_id
)) group by o_order_id
) t3 on t1.o_order_id = t3.o_order_id);