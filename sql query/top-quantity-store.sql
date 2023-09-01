/*Top Total Quantity from Store*/
select t.storeid as store_id, s.storename as store_name, sum(t.qty) as quantity
from "transaction" t
join store s 
on t.storeid = s.storeid 
group by t.storeid, s.storename
order by sum(t.qty) desc 
