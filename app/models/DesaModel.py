from app.extension import db

class Desa(db.Model):
    __tablename__ = 'daftar_desa'

    id_desa = db.Column(db.Integer, primary_key=True)
    id_kecamatan = db.Column(db.Integer, db.ForeignKey('daftar_kecamatan.id'), nullable=False)
    nama_desa = db.Column(db.String(100), nullable=False)

    kecamatan = db.relationship('Kecamatan', backref=db.backref('desa_list', lazy=True))
