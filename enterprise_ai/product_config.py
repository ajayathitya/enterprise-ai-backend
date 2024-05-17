from langchain_core.pydantic_v1 import BaseModel

class ProductConfig(BaseModel):
    org_name: str = "Post Consumer Brands"
    org_id: str = "postconsumerbrands"
    about: str = "Post Consumer Brands produce iconic breakfast cereals, snacks and pet food. "
    catalog: str = "Post Consumer Brands sells only the following iconic breakfast cereals, snacks and pet food. The brands under PCB include: Alpen Muesli, Barbara's, Better Oats, Bran Flakes, Coco Wheats, Disney100, Farina Mills, Golden Crisp, Grape-Nuts, Great Grains, Honey Bunches of Oats, Honey Maid S’mores, Honeycomb, Malt-O-Meal Hot, Malt-O-Meal, Mom's Best, Honey Ohs!, Oreo O’s, Pebbles, Premier Protein, Puffins, Raisin Bran, Shredded Wheat, Snoop Cereal, Sweet Dreams, Sweet Home Farm, Uncle Sam, Waffle Crisp, Weetabix."
    contact_info: dict = {
        "phone_number":"1-800-431-7678",
        "time":"9AM - 5PM",
        "days": "Mon - Fri",
        "time_zone": "ET"
    }
    topics: dict = {
        "query":"product details, pricing, ingredients, usage",
        "complaint": "quality concerns, shipment delays",
        "order": "create order, update order, cancel order, view order status"
    }
product_config = ProductConfig()