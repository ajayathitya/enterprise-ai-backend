from langchain_core.pydantic_v1 import BaseModel

# class ProductConfig(BaseModel):
#     org_name: str = "Post Consumer Brands"
#     org_id: str = "postconsumerbrands"
#     about: str = "Post Consumer Brands produce iconic breakfast cereals, snacks and pet food. "
#     catalog: str = "Post Consumer Brands sells only the following iconic breakfast cereals, snacks and pet food. The brands under PCB include: Alpen Muesli, Barbara's, Better Oats, Bran Flakes, Coco Wheats, Disney100, Farina Mills, Golden Crisp, Grape-Nuts, Great Grains, Honey Bunches of Oats, Honey Maid S’mores, Honeycomb, Malt-O-Meal Hot, Malt-O-Meal, Mom's Best, Honey Ohs!, Oreo O’s, Pebbles, Premier Protein, Puffins, Raisin Bran, Shredded Wheat, Snoop Cereal, Sweet Dreams, Sweet Home Farm, Uncle Sam, Waffle Crisp, Weetabix."
#     contact_info: dict = {
#         "phone_number":"1-800-431-7678",
#         "time":"9AM - 5PM",
#         "days": "Mon - Fri",
#         "time_zone": "ET"
#     }
#     topics: dict = {
#         "query":"product details, pricing, ingredients, usage",
#         "complaint": "quality concerns, shipment delays",
#         "order": "create order, update order, cancel order, view order status"
#     }
# product_config = ProductConfig()

class ProductConfig(BaseModel):
    org_name: str = "Kate Farms"
    org_id: str = "katefarms"
    about: str = "Kate Farms produces nutritional shakes with plant-based ingredients, including USDA Organic pea protein, to provide the high-quality nutrition. ALl of their products are dairy-free, gluten-free, designed for easy digestion, nutritionally complete and delicious in taste."
    catalog: str = "Kate Farms sells only the following products: Everyday Nutrition Products - Kids Nutrition Shake, Nutrition Shake. Medical Nutrition - Standard 1.0, Standard 1.4, Peptide 1.0, Pepptide 1.5. Specialized Nutrition - Glucose Support 1.2, Renal Support 1.8. Kids Nutrition - Pediatric Standard 1.2, Pediatric Peptide 1.0, Pediatric Peptide 1.5"
    contact_info: dict = {
        "phone_number":"1-805-845-2446",
        "time":"8AM - 8PM",
        "days": "Mon - Fri",
        "time_zone": "ET"
    }
    topics: dict = {
        "query":"product details, pricing, ingredients, usage",
        "complaint": "quality concerns, shipment delays",
        "order": "create order, update order, cancel order, view order status"
    }
product_config = ProductConfig()