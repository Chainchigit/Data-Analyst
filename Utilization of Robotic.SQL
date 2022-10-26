ALTER TABLE SP26_10_22 ADD station;

SELECT
    description,
    last,
    station,
    sprayer,
    total_spray
FROM Spray 
WHERE station <> "NULL"
ORDER By last ;

UPDATE Spray
SET station ='Gaiotto'
WHERE sprayer IN("10","17","22","33","51","42","67","29","36","59","16","63",
                "O10","O17","O22","O33","O51","O42","O67","O29","O36","O59","O16","O63") ;

UPDATE Spray
SET station ='Motoman'
WHERE sprayer IN("26","23","43","41","13",
                "O26","O23","O43","O41","O13");
                
UPDATE Spray
SET station ='Manual'
WHERE sprayer IN("4","18","11","99","66","52",
                "O4","O18","O11","O99","O66","O52")
