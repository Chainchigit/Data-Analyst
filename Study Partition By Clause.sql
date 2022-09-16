#PARTITION BY Clause in Action
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
#Calculate the Moving Average
SELECT
  shop,
  date,
  revenue_amount,
  AVG(revenue_amount) OVER (
    PARTITION BY shop
    ORDER BY date ASC
    RANGE BETWEEN INTERVAL '1' DAY PRECEDING AND CURRENT ROW
  ) AS moving_avg
FROM revenue_per_shop;

---------------------------------#4-------------------------------------------
#Moving Average for Databases That Don't Support Using RANGE with Date/Time Data Types
SELECT
  shop,
  date,
  revenue_amount,
  date - '2021_05_01' AS day_difference,
  AVG(revenue_amount) OVER (
    PARTITION BY shop
    ORDER BY (date - '2021_05_01')
    RANGE BETWEEN 1 PRECEDING AND CURRENT ROW
  ) AS moving_avg
FROM revenue_per_shop;

---------------------------------#5-------------------------------------------
#Find the Last Value Within a Range
SELECT
  shop,
  date,
  revenue_amount,
  LAST_VALUE(revenue_amount) OVER (
    PARTITION BY shop
    ORDER BY date
    RANGE BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING
  ) AS last_value
FROM revenue_per_shop;

--------------------------------#6---------------------------------------------
#Find the Number of Items Within a Range
SELECT
  shop,
  date,
  revenue_amount,
  COUNT(*) OVER (
    ORDER BY revenue_amount ASC
    RANGE BETWEEN 1000 PRECEDING AND 1000 FOLLOWING
  ) AS number_of_days
FROM revenue_per_shop;

-------------------------------#7----------------------------------------------
# Find the Maximum Value
SELECT
  shop,
  date,
  revenue_amount,
  MAX(revenue_amount) OVER (
    ORDER BY DATE
    RANGE BETWEEN INTERVAL '3' DAY PRECEDING AND INTERVAL '1' DAY FOLLOWING
  ) AS max_revenue
FROM revenue_per_shop;

------------------------------#8-----------------------------------------------

