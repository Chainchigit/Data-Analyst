-- Define menu from price compare to customersegments

SELECT 
	A.customersegmentsId,
    A.title,
    B.name,
    B.unitprice
from customersegments AS A
JOIN menu AS B
ON A.customersegmentsId = B.customersegmentsId
JOIN invoice AS C
ON B.menuId = C.menuId
WHERE B.unitprice IN (40,50,80) 
GROUP By 3
ORDER BY 4 DESC;
