import json
from sqlalchemy import create_engine, Column, Integer, String, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

# Define the base class for SQLAlchemy models
Base = declarative_base()

# Define the DiseaseMaster table model
class DiseaseMaster(Base):
    __tablename__ = 'disease_master'

    id = Column(Integer, primary_key=True)
    disease_name = Column(String, nullable=False)
    icd11_code = Column(String)
    org_id = Column(String)
    active_flag = Column(String)  # Using String since JSON shows 'Y'/'N'
    country_code = Column(String)
    snowmed_ct = Column(String, nullable=True)  # Allowing NULL as per JSON

# Create SQLite database and table
engine = create_engine('sqlite:///voice_ai.db', echo=True)
Base.metadata.create_all(engine)

# Create a session
Session = sessionmaker(bind=engine)
session = Session()

# Read JSON file
with open('disease_master.json', 'r') as file:
    data = json.load(file)

# Insert data into the table
for entry in data['zh_disease_master']:
    disease = DiseaseMaster(
        id=entry['id'],
        disease_name=entry['disease_name'],
        icd11_code=entry['icd11_code'],
        org_id=entry['org_id'],
        active_flag=entry['active_flag'],
        country_code=entry['country_code'],
        snowmed_ct=entry['snowmed_ct']
    )
    session.add(disease)

# Commit the transaction
session.commit()

# Close the session
session.close()