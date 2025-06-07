from app.extension import db

class Cluster(db.Model):
    __tablename__ = 'cluster_perkebunan'

    id_cluster = db.Column(db.Integer, primary_key=True)
    id_desa = db.Column(db.Integer, db.ForeignKey('daftar_desa.id_desa'), nullable=False)
    nama_cluster = db.Column(db.String(50), nullable=False)
    deskripsi = db.Column(db.Text)

    desa = db.relationship('Desa', backref=db.backref('cluster', lazy=True))