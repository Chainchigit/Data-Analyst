-- SubQurry AVG invoice From Europe  
WITH europe AS (	
	SELECT * FROM customers
	WHERE country IN ('Portugal','Germany','Italy')
	
),	invoice_2012 AS (
	SELECT * FROM invoices
	WHERE STRFTIME("%Y", invoicedate) = '2012'
)

SELECT EU.FirstName,
	   Country, 
	   AVG(inv.total)  
FROM europe AS EU
JOIN invoice_2012 AS inv
ON EU.customerid = inv.customerid
GROUP BY firstname 
ORDER BY 3 DESC;
