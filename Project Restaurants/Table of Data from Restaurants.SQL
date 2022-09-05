-- Create table farmer_ingredients
Create table farmer_ingredients (
    farmer_ingredientsId INT NOT NULL PRIMARY KEY ,
    title TEXT,
    name TEXT,
    province TEXT,
    phone NVARCHAR(12),
    email NVARCHAR
);
-- Insert table farmer_ingredients
Insert into farmer_ingredients values
    (1, "Uni Fram", "Satoshi Nakamoto ","Surat Thani", 
    "081-100-1001", "SatoshiN@gmail.com"),
    (2, "Pork Fram", "Vitalik Buterin","Chiang Mai", 
    "081-200-2002", "VitalikB@gmail.com"),
    (3, "Rice Fram", "Abraham Lincoln ","Surin", 
    "081-300-3003", "AbrahamL@gmail.com"),
    (4, "Fish Fram", "Donald Trump ","Chiang Rai",
    "081-400-4004", "DonaldT@gmail.com"),
    (5, "Organic Salad Fram", "Muhammad Ali", "Udon Thani",
    "081-500-5005", "MuhammadA@gmail.com"),
    (6, "Cow Fram", "Charles Darwin", "Lop Buri", 
    "081-600-6006", "CharlesD@gmail.com"),
    (7, "Scallop Farm", "Paul McCartney", "Surat Thani",
    "081-700-7007", "PaulM@gmail.com"),
    (8, "Strawberry Fram", "Vladimir Putin", "Chiang Mai", 
    "081-800-8008", "VladimirP@gmail.com"),
    (9, "Fruity Fram", "J.K.Rowling ", "Chiang Rai", 
    "081-900-9009", "J.K.Rowling@gmail.com"),
    (10,"Durian Fram", "Peter Sellers ", "Rayong", 
    "081-110-1010", "PeterS@gmail.com");

-- Create table supermarkets
Create table supermarkets (
    supermarketsId INT NOT NULL PRIMARY KEY ,
    title TEXT,
    phone NVARCHAR(12),
    email NVARCHAR
);
-- Insert table supermarkets
Insert into supermarkets values
    (01, "Tesco Lotus",       "082-100-1001", "Tesco@gmail.com"),
    (02, "Big C Supercenter", "082-200-1002", "BigC@gmail.com"),
    (03, "Makro",             "082-300-1003", "Makro@gmail.com"),
    (04, "Top Supermarkets",  "082-400-1004", "Top@gmail.com"),
    (05, "Villa Market",      "082-500-1005", "Vila@gmail.com"),
    (06, "Gourmet Market",    "082-600-1006", "Gourmet@gmail.com"),
    (07, "Max Valu",          "082-700-1007", "MaxValu@gmail.com"),
    (08, "Isetan Supermarket","082-800-1008", "Isetan@gmail.com"),
    (09, "CP Freshmart",      "082-900-1009", "CPfresh@gmail.com"),
    (10, "Betago",            "082-900-1009", "Betago@gmail.com");

-- Create table staff--
Create table staff (
    staffId INT NOT NULL PRIMARY KEY ,
    name TEXT,
    job title TEXT,
    phone NVARCHAR(12),
    email NVARCHAR,
    reportTo INT
);
-- Insert table staff
Insert into staff values 
    (001,"Steve Jobs " , "Manager", "083-100-1001", "SteveJN@gmail.com", 1),
    (002,"Ronald Reagan","Chef",  "083-200-1002", "RonaldR@gmail.com", 01),
    (003,"Greta Thunberg ","Bill Service", "083-300-1003", "GretaT@gmail.com", 01),
    (004,"Ronald Reagan","Asistance Chef", "083-400-1004", "RonaldR@gmail.com", 02),
    (005,"Plato ","Waiter",  "083-500-1005", "Plato@gmail.com", 03),
    (006,"Plat ","Waiter", "083-500-1005", "Plat@gmail.com", 03);

-- Create table customersegments
Create table customersegments (
    customersegmentsId INT NOT NULL PRIMARY KEY,
    title TEXT
);

-- Insert table customersegments
Insert into customersegments values 
    (1, "Vegetarian and Gluten intolerent"),
    (2, "Employee on lunch breaks"),
    (3, "Helth conscions"),
    (4, "Sushi lovers"),
    (5, "General public/passers by");

-- Create table menu
Create table menu (
    menuId INT NOT NULL PRIMARY KEY,
    name TEXT,
    unitPrice REAL,
    customersegmentsId INT NOT NULL,
    FOREIGN KEY (customersegmentsId) REFERENCES customersegments(customersegmentsId)

);
-- Insert table menu
Insert into menu values
    (01, "Sushi", 40, 4),
    (02, "Sashimi", 50, 4),
    (03, "Unagi Grilled Eel", 65, 2),
    (04, "Tempura", 80, 1),
    (05, "Soba (Buckwheat Noodles) and Udon (Wheat Noodles)", 80, 5),
    (06, "Onigiri Rice Balls", 75, 1),
    (07, "Yakitori Grilled Chicken Skewers", 110, 5),
    (08, "Sukiyaki", 85, 3),
    (09, "Oden Simmered Ingredients", 80, 5),
    (10, "Miso Soup", 55, 3);

-- Create table invoice
Create table invoice (
    invoiceId INT NOT NULL PRIMARY KEY,
    invoice_date text,
    menuId INT,
    farmer_ingredientsId INT,
    supermarketsId INT,    
    staffId INT,
    customersegmentsId INT,
    unitPrice REAL,
    Total REAL,
    FOREIGN KEY (menuId) REFERENCES menu(menuId),
    FOREIGN KEY (unitPrice) REFERENCES menu(unitPrice)
);
INSERT INTO invoice values
    (1,'2022-05-04',  01,1,01,003,4,40,160),
    (2,'2022-05-04',  01,7,08,003,4,40,160),
    (3,'2022-05-04',  02,1,07,003,4,50,200),
    (4,'2022-05-04',  04,7,06,003,1,80,80),
    (5,'2022-05-04',  05,2,08,003,5,80,400),
    (6,'2022-05-04',  05,2,10,003,5,80,400),
    (7,'2022-05-04',  05,6,02,003,5,80,400),
    (8,'2022-05-04',  08,5,02,003,3,85,255),
    (9,'2022-05-04',  09,3,03,003,5,80,400),
    (10,'2022-05-04', 10,5,04,003,3,55,165);
