from app.extension import db

class Perkebunan(db.Model):
    __tablename__ = 'data_perkebunan'

    id_perkebunan = db.Column(db.Integer, primary_key=True)
    id_kecamatan = db.Column(db.Integer, db.ForeignKey('daftar_kecamatan.id'), nullable=False)
    id_desa = db.Column(db.Integer, db.ForeignKey('daftar_desa.id_desa'), nullable=False)

    luas_tbm = db.Column(db.Numeric(10, 2))
    luas_tm = db.Column(db.Numeric(10, 2))
    luas_ttm = db.Column(db.Numeric(10, 2))
    luas_jumlah = db.Column(db.Numeric(10, 2))
    produksi_ton = db.Column(db.Numeric(10, 2))
    produktivitas_kg_ha = db.Column(db.Numeric(10, 2))
    jumlah_petani_kk = db.Column(db.Integer)

    kecamatan = db.relationship('Kecamatan', backref=db.backref('perkebunan_list', lazy=True))
    desa = db.relationship('Desa', backref=db.backref('perkebunan_list', lazy=True))
