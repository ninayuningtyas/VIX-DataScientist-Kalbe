/*Top Total Amount from Product*/
select t.productid as product_id, 
	   p."Product Name" as product_name, 
	   SUM(t.totalamount) as total_amount
from "transaction" t 
join product p 
on t.productid = p.productid 
group by t.productid, p."Product Name"
order by SUM(t.totalamount) desc 