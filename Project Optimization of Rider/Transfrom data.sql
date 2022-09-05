Select  rider_name, 
	UTC_Date, 
        UTC_Hour, count(*) as Total_Order
from (

       select rider_name,
  	      date(created_date,'+7 hours') as UTC_DATE, 
  	      strftime('%H',TIME(created_date,'+7 hours') ) as UTC_Hour from rider 
       where status <> 'Cancel'

  ) as Rider_Report
  
 Group by rider_name,UTC_Date, UTC_HOUR
 order by rider_name,UTC_Date, UTC_HOUR;
