from app.routes.AuthRoute import auth_bp
from app.routes.DashboardRoute import dashboard_bp
from app.routes.KecamatanRoute import kecamatan_bp
from app.routes.DesaRoute import desa_bp
from app.routes.PerkebunanRoute import perkebunan_bp
from app.routes.ClusterRoute import cluster_bp
from app.routes.GMMRoute import gmm_bp

def register_routes(app):
    app.register_blueprint(auth_bp)
    app.register_blueprint(dashboard_bp)
    app.register_blueprint(kecamatan_bp)
    app.register_blueprint(desa_bp)
    app.register_blueprint(perkebunan_bp)
    app.register_blueprint(cluster_bp)
    app.register_blueprint(gmm_bp)