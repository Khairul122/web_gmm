import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    SECRET_KEY = os.getenv('SECRET_KEY', 'default_dev_key')
    SQLALCHEMY_DATABASE_URI = 'mysql+pymysql://root:@localhost/db_gmm'
    SQLALCHEMY_TRACK_MODIFICATIONS = False
