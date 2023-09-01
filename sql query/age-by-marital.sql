/*Rename column*/
ALTER TABLE customers 
RENAME COLUMN "Marital Status" TO marital_status;

/*Fill the null value with "Unidentified*/
UPDATE "vix-kalbe".customers 
SET marital_status = 'Unknown'
WHERE marital_status IS NULL;

/*Average Cust Age based on Marital Status*/
select round(avg(age),2) as "Age Average", marital_status  
from customers c 
where marital_status is not null 
group by marital_status;
