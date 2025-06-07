from app.extension import db

class Kecamatan(db.Model):
    __tablename__ = 'daftar_kecamatan'

    id = db.Column(db.Integer, primary_key=True)
    nama_kecamatan = db.Column(db.String(100), unique=True, nullable=False)
