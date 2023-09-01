/*Update the value
 * 0 as Female, 1 as Male*/
-- Alter the datatype of the column
ALTER TABLE customers 
ALTER COLUMN gender TYPE VARCHAR;

-- Update the values
UPDATE customers
SET gender = CASE 
    WHEN gender = '0' THEN 'Female'
    WHEN gender = '1' THEN 'Male'
    ELSE gender
    END;

/*Age Average by Gender*/
select round(avg(age),2) as "Age Average", gender  
from customers c 
group by gender;