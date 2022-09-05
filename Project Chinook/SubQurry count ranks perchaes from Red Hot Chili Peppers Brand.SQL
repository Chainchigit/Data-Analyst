-- SubQurry count ranks perchaes from Red Hot Chili Peppers Brand
SELECT 
	city || ' ' || country AS Area,
	COUNT(trackname) AS purchase,
	artistname
FROM (
SELECT 
	customers.city,
	customers.country,
	invoices.invoicedate,
	artists.name AS artistname,
	albums.title,
	tracks.name AS trackname,
	invoice_items.UnitPrice
FROM customers
JOIN invoices ON customers.CustomerId = invoices.CustomerId
JOIN invoice_items ON invoices.InvoiceId = invoice_items.InvoiceId
JOIN tracks ON tracks.trackid = invoice_items.trackid
JOIN albums ON tracks.albumid = albums.albumid
JOIN artists ON  artists.artistid = albums.artistid)
WHERE artistname = 'Red Hot Chili Peppers'
GROUP BY 1
ORDER BY 2 DESC;
