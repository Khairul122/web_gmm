from app.extension import db

class GMM(db.Model):
    __tablename__ = 'gmm'

    id_clustering = db.Column(db.Integer, primary_key=True)
    id_desa = db.Column(db.Integer, db.ForeignKey('daftar_desa.id_desa'), nullable=False)
    id_cluster = db.Column(db.Integer, db.ForeignKey('cluster_perkebunan.id_cluster'), nullable=False)
    probabilitas = db.Column(db.Float)
    mean_vector = db.Column(db.Text)
    covariance_matrix = db.Column(db.Text)
    iteration = db.Column(db.Integer)
    converged = db.Column(db.Boolean)
    silhouette_score = db.Column(db.Float)         
    davies_bouldin_index = db.Column(db.Float)

    desa = db.relationship('Desa', backref=db.backref('gmm_clustering', lazy=True))
    cluster = db.relationship('Cluster', backref=db.backref('gmm_clustering', lazy=True))