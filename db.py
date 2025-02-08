from pymongo import MongoClient
import os
from dotenv import load_dotenv

load_dotenv()

MONGOURI = os.getenv("MONGO_URI")

client = MongoClient(MONGOURI)
db = client.QADB