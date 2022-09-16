SELECT DISTINCT
       flight_number,
       aircraft_model,
    SUM(num_of_passengers) OVER (PARTITION BY flight_number, aircraft_model)
                                                            AS total_passengers,
    SUM(total_revenue) OVER (PARTITION BY flight_number, aircraft_model)
                                                            AS total_revenue
FROM paris_london_flights
ORDER BY flight_number, aircraft_model;

----------------------------------#1------------------------------------

WITH year_month_data AS (
  SELECT DISTINCT
       EXTRACT(YEAR FROM scheduled_departure) AS year,
       EXTRACT(MONTH FROM scheduled_departure) AS month,
       SUM(number_of_passengers)
              OVER (PARTITION BY EXTRACT(YEAR FROM scheduled_departure),
                                  EXTRACT(MONTH FROM scheduled_departure)
                   ) AS passengers
   FROM  paris_london_flights
  ORDER BY 1, 2
)
SELECT  year,
        month,
     passengers,
     LAG(passengers) OVER (ORDER BY year, month) passengers_previous_month,
     passengers - LAG(passengers) OVER (ORDER BY year, month) AS passengers_delta
FROM year_month_data;

--------------------------------------#2---------------------------------

WITH paris_london_delays AS (
  SELECT DISTINCT
       aircraft_model,
       EXTRACT(YEAR FROM scheduled_departure) AS year,
       EXTRACT(MONTH FROM scheduled_departure) AS month,
       AVG(real_departure - scheduled_departure) AS month_delay
   FROM  paris_london_flights
   GROUP BY 1, 2, 3
)
SELECT  DISTINCT
     aircraft_model,
     year,
     month,
     month_delay AS monthly_avg_delay,
     AVG(month_delay) OVER (PARTITION BY aircraft_model, year) AS year_avg_delay,
     AVG(month_delay) OVER (PARTITION BY year) AS year_avg_delay_all_models,
     AVG(month_delay) OVER (PARTITION BY aircraft_model, year
                               ORDER BY month
                               ROWS BETWEEN 3 PRECEDING AND CURRENT ROW
                            ) AS rolling_average_last_4_months
 
FROM paris_london_delays
ORDER BY 1,2,3

--------------------------------#3--------------------------------------------
