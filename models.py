import sqlalchemy as db
from sqlalchemy.orm import declarative_base

Base = declarative_base()

class Event(Base):
    __tablename__   = "events"
    id              = db.Column(db.Integer, primary_key=True, index=True, autoincrement=True)
    name            = db.Column(db.String)
    location        = db.Column(db.String)
    category        = db.Column(db.String)
    image_url       = db.Column(db.String)
    description     = db.Column(db.Text)
    start_date      = db.Column(db.Date)
    end_date        = db.Column(db.Date)
    created_at      = db.Column(db.DateTime)
    updated_at      = db.Column(db.DateTime)
    
    @property
    def serialize(self):
        return {
            "id": self.id,
            "name": self.name,
            "location": self.location,
            "category": self.category,
            "image_url": self.image_url,
            "description": self.description,
            "start_date": self.start_date,
            "end_date": self.end_date,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }