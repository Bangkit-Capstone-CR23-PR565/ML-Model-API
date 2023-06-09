import sqlalchemy as db
from sqlalchemy.orm import declarative_base

Base = declarative_base()

class User(Base):
    __tablename__       = "users"
    id                  = db.Column(db.Integer, primary_key=True, index=True, autoincrement=True)
    email               = db.Column(db.String)
    phone               = db.Column(db.String)
    password            = db.Column(db.String)
    full_name           = db.Column(db.String)
    location            = db.Column(db.String)
    category_interest   = db.Column(db.String)
    refresh_token       = db.Column(db.String)
    created_at          = db.Column(db.DateTime)
    updated_at          = db.Column(db.DateTime)

    @property
    def serialize(self):
        return {
            "id"                  : self.id,
            "email"               : self.email,
            "phone"               : self.phone,
            "password"            : self.password,
            "full_name"           : self.full_name,
            "location"            : self.location,
            "category_interest"   : self.category_interest,
            "refresh_token"       : self.refresh_token,
            "created_at"          : self.created_at,
            "updated_at"          : self.updated_at,
        }

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
            "id"            : self.id,
            "name"          : self.name,
            "location"      : self.location,
            "category"      : self.category,
            "image_url"     : self.image_url,
            "description"   : self.description,
            "start_date"    : self.start_date,
            "end_date"      : self.end_date,
            "created_at"    : self.created_at,
            "updated_at"    : self.updated_at,
        }

class Rating(Base):
    __tablename__   = 'rating'
    id              = db.Column(db.Integer, primary_key=True, index=True, autoincrement=True, nullable=False)
    user_id         = db.Column(db.Integer)
    event_id        = db.Column(db.Integer)
    user_rating     = db.Column(db.Integer)
    user_comment    = db.Column(db.Text)
    created_at      = db.Column(db.DateTime)
    updated_at      = db.Column(db.DateTime)

    @property
    def serialize(self):
        return {
            "id"            : self.id,
            "user_id"       : self.user_id,
            "event_id"      : self.event_id,
            "user_rating"   : self.user_rating,
            "user_comment"  : self.user_comment,
            "created_at"    : self.created_at,
            "updated_at"    : self.updated_at,
        }