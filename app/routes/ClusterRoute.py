from flask import Blueprint, render_template, request, redirect, url_for, flash
from app.models.ClusterModel import Cluster
from app.models.DesaModel import Desa
from app.extension import db
import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from app.models.PerkebunanModel import Perkebunan as DataPerkebunan
import warnings
warnings.filterwarnings('ignore')

cluster_bp = Blueprint('cluster', __name__, url_prefix='/cluster')

@cluster_bp.route('/')
def index():
    try:
        data = Cluster.query.join(Desa).order_by(Cluster.nama_cluster.asc()).all()
        
        cluster_stats = {
            'tinggi': 0,
            'menengah': 0,
            'rendah': 0,
            'total': len(data)
        }
        
        for item in data:
            if 'Produksi Tinggi' in item.nama_cluster:
                cluster_stats['tinggi'] += 1
            elif 'Produksi Menengah' in item.nama_cluster:
                cluster_stats['menengah'] += 1
            elif 'Produksi Rendah' in item.nama_cluster:
                cluster_stats['rendah'] += 1
        
        return render_template('cluster/index.html', data=data, stats=cluster_stats)
    except Exception as e:
        flash(f'Error loading cluster data: {str(e)}', 'error')
        return render_template('cluster/index.html', data=[], stats={'tinggi': 0, 'menengah': 0, 'rendah': 0, 'total': 0})

@cluster_bp.route('/hapus/<int:id>')
def hapus(id):
    try:
        cluster = Cluster.query.get_or_404(id)
        
        dependent_count = db.session.execute(
            'SELECT COUNT(*) as count FROM gmm WHERE id_cluster = :id_cluster', 
            {'id_cluster': id}
        ).fetchone()
        
        if dependent_count and dependent_count[0] > 0:
            db.session.execute('DELETE FROM gmm WHERE id_cluster = :id_cluster', {'id_cluster': id})
            flash(f'Menghapus {dependent_count[0]} data GMM terkait cluster', 'info')
        
        db.session.delete(cluster)
        db.session.commit()
        flash('Data cluster berhasil dihapus', 'success')
        
    except Exception as e:
        db.session.rollback()
        flash(f'Error saat menghapus data: {str(e)}', 'error')
    
    return redirect(url_for('cluster.index'))

@cluster_bp.route('/hapus-semua')
def hapus_semua():
    try:
        total_cluster = Cluster.query.count()
        
        if total_cluster == 0:
            flash('Tidak ada data cluster untuk dihapus', 'warning')
            return redirect(url_for('cluster.index'))
        
        dependent_count = db.session.execute('SELECT COUNT(*) as count FROM gmm').fetchone()
        total_gmm = dependent_count[0] if dependent_count else 0
        
        if total_gmm > 0:
            db.session.execute('DELETE FROM gmm')
            flash(f'Menghapus {total_gmm} data GMM terkait', 'info')
        
        Cluster.query.delete()
        db.session.commit()
        
        flash(f'Berhasil menghapus {total_cluster} data cluster dan {total_gmm} data GMM terkait', 'success')
        
    except Exception as e:
        db.session.rollback()
        flash(f'Error saat menghapus semua data: {str(e)}', 'error')
    
    return redirect(url_for('cluster.index'))

def calculate_bic_score(X, n_components, covariance_type='full', max_iter=200, n_init=10):
    try:
        gmm = GaussianMixture(
            n_components=n_components,
            covariance_type=covariance_type,
            max_iter=max_iter,
            n_init=n_init,
            random_state=42
        )
        gmm.fit(X)
        
        bic_score = gmm.bic(X)
        aic_score = gmm.aic(X)
        log_likelihood = gmm.score(X)
        
        if n_components > 1:
            labels = gmm.predict(X)
            silhouette = silhouette_score(X, labels)
        else:
            silhouette = 0
            
        return {
            'bic': bic_score,
            'aic': aic_score,
            'log_likelihood': log_likelihood,
            'silhouette': silhouette,
            'model': gmm
        }
    except Exception as e:
        return None

def find_optimal_clusters(X, max_clusters=8):
    results = []
    
    for n_clusters in range(1, max_clusters + 1):
        result = calculate_bic_score(X, n_clusters)
        if result:
            results.append({
                'n_clusters': n_clusters,
                'bic': result['bic'],
                'aic': result['aic'],
                'log_likelihood': result['log_likelihood'],
                'silhouette': result['silhouette'],
                'model': result['model']
            })
    
    if not results:
        return None
    
    best_bic = min(results, key=lambda x: x['bic'])
    best_aic = min(results, key=lambda x: x['aic'])
    
    combined_score = []
    multi_cluster_results = [r for r in results if r['n_clusters'] > 1]
    
    if multi_cluster_results:
        bic_values = [r['bic'] for r in multi_cluster_results]
        silhouette_values = [r['silhouette'] for r in multi_cluster_results]
        
        if max(bic_values) != min(bic_values) and max(silhouette_values) != min(silhouette_values):
            for result in multi_cluster_results:
                norm_bic = (result['bic'] - min(bic_values)) / (max(bic_values) - min(bic_values))
                norm_silhouette = (result['silhouette'] - min(silhouette_values)) / (max(silhouette_values) - min(silhouette_values))
                
                score = 0.7 * (1 - norm_bic) + 0.3 * norm_silhouette
                combined_score.append((result, score))
    
    if combined_score:
        best_combined = max(combined_score, key=lambda x: x[1])[0]
    else:
        best_combined = best_bic
    
    return {
        'best_model': best_combined,
        'all_results': results,
        'best_bic': best_bic,
        'best_aic': best_aic
    }

def assign_cluster_names(gmm_model, X_scaled, data_original):
    labels = gmm_model.predict(X_scaled)
    centers = gmm_model.means_
    
    cluster_stats = {}
    for i in range(len(centers)):
        cluster_mask = labels == i
        cluster_data = data_original[cluster_mask]
        
        if len(cluster_data) > 0:
            avg_produksi = np.mean(cluster_data[:, 3])
            avg_luas_tm = np.mean(cluster_data[:, 1])
            avg_petani = np.mean(cluster_data[:, 4])
            
            cluster_stats[i] = {
                'avg_produksi': avg_produksi,
                'avg_luas_tm': avg_luas_tm,
                'avg_petani': avg_petani,
                'count': len(cluster_data),
                'center': centers[i]
            }
    
    sorted_clusters = sorted(cluster_stats.items(), key=lambda x: x[1]['avg_produksi'])
    
    cluster_mapping = {}
    cluster_names = ['Produksi Rendah', 'Produksi Menengah', 'Produksi Tinggi']
    
    for idx, (cluster_id, stats) in enumerate(sorted_clusters):
        if idx < len(cluster_names):
            cluster_mapping[cluster_id] = {
                'name': cluster_names[idx],
                'description': f"Luas TBM rata-rata: {np.mean([data_original[labels == cluster_id, 0]]):.2f} Ha, "
                             f"Luas TM rata-rata: {stats['avg_luas_tm']:.2f} Ha, "
                             f"Luas TTM rata-rata: {np.mean([data_original[labels == cluster_id, 2]]):.2f} Ha, "
                             f"Produksi rata-rata: {stats['avg_produksi']:.2f} Ton, "
                             f"Jumlah petani: ±{int(stats['avg_petani'])} KK"
            }
        else:
            if stats['avg_produksi'] <= sorted_clusters[0][1]['avg_produksi']:
                cluster_mapping[cluster_id] = {
                    'name': 'Produksi Rendah',
                    'description': f"Luas TBM rata-rata: {np.mean([data_original[labels == cluster_id, 0]]):.2f} Ha, "
                                 f"Luas TM rata-rata: {stats['avg_luas_tm']:.2f} Ha, "
                                 f"Luas TTM rata-rata: {np.mean([data_original[labels == cluster_id, 2]]):.2f} Ha, "
                                 f"Produksi rata-rata: {stats['avg_produksi']:.2f} Ton, "
                                 f"Jumlah petani: ±{int(stats['avg_petani'])} KK"
                }
            elif stats['avg_produksi'] >= sorted_clusters[-1][1]['avg_produksi']:
                cluster_mapping[cluster_id] = {
                    'name': 'Produksi Tinggi',
                    'description': f"Luas TBM rata-rata: {np.mean([data_original[labels == cluster_id, 0]]):.2f} Ha, "
                                 f"Luas TM rata-rata: {stats['avg_luas_tm']:.2f} Ha, "
                                 f"Luas TTM rata-rata: {np.mean([data_original[labels == cluster_id, 2]]):.2f} Ha, "
                                 f"Produksi rata-rata: {stats['avg_produksi']:.2f} Ton, "
                                 f"Jumlah petani: ±{int(stats['avg_petani'])} KK"
                }
            else:
                cluster_mapping[cluster_id] = {
                    'name': 'Produksi Menengah',
                    'description': f"Luas TBM rata-rata: {np.mean([data_original[labels == cluster_id, 0]]):.2f} Ha, "
                                 f"Luas TM rata-rata: {stats['avg_luas_tm']:.2f} Ha, "
                                 f"Luas TTM rata-rata: {np.mean([data_original[labels == cluster_id, 2]]):.2f} Ha, "
                                 f"Produksi rata-rata: {stats['avg_produksi']:.2f} Ton, "
                                 f"Jumlah petani: ±{int(stats['avg_petani'])} KK"
                }
    
    return cluster_mapping, labels

@cluster_bp.route('/analisis-bic')
def analisis_bic():
    try:   
        data_perkebunan = DataPerkebunan.query.all()
        
        if not data_perkebunan:
            flash('Tidak ada data perkebunan untuk dianalisis', 'error')
            return redirect(url_for('cluster.index'))
        
        if len(data_perkebunan) < 3:
            flash('Data terlalu sedikit untuk analisis clustering yang optimal', 'warning')
            return redirect(url_for('cluster.index'))
        
        try:
            db.session.execute('DELETE FROM gmm')
            Cluster.query.delete()
            db.session.commit()
        except Exception as clear_error:
            db.session.rollback()
        
        X = []
        desa_ids = []
        
        for data in data_perkebunan:
            luas_tbm = float(data.luas_tbm or 0)
            luas_tm = float(data.luas_tm or 0)
            luas_ttm = float(data.luas_ttm or 0)
            produksi_ton = float(data.produksi_ton or 0)
            jumlah_petani_kk = int(data.jumlah_petani_kk or 0)
            
            X.append([luas_tbm, luas_tm, luas_ttm, produksi_ton, jumlah_petani_kk])
            desa_ids.append(data.id_desa)
        
        X = np.array(X)
        X_original = X.copy()
        
        if np.any(np.isnan(X)) or np.any(np.isinf(X)):
            X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        max_clusters = min(3, len(data_perkebunan) - 1)
        optimization_result = find_optimal_clusters(X_scaled, max_clusters)
        
        if not optimization_result:
            flash('Gagal melakukan optimasi clustering', 'error')
            return redirect(url_for('cluster.index'))
        
        best_model = optimization_result['best_model']['model']
        n_clusters = optimization_result['best_model']['n_clusters']
        bic_score = optimization_result['best_model']['bic']
        silhouette = optimization_result['best_model']['silhouette']
        
        cluster_mapping, labels = assign_cluster_names(best_model, X_scaled, X_original)
        
        cluster_counts = {}
        successful_saves = 0
        
        for i, data in enumerate(data_perkebunan):
            try:
                cluster_id = labels[i]
                cluster_info = cluster_mapping[cluster_id]
                
                cluster = Cluster(
                    id_desa=data.id_desa,
                    nama_cluster=cluster_info['name'],
                    deskripsi=cluster_info['description']
                )
                db.session.add(cluster)
                
                if cluster_info['name'] not in cluster_counts:
                    cluster_counts[cluster_info['name']] = 0
                cluster_counts[cluster_info['name']] += 1
                successful_saves += 1
                
            except Exception as save_error:
                continue
        
        if successful_saves > 0:
            db.session.commit()
            cluster_summary = ', '.join([f"{name} ({count} desa)" for name, count in cluster_counts.items()])
            flash(f'Penentuan Jumlah Cluster Optimal berhasil! Optimal clusters: {n_clusters} dengan BIC score: {bic_score:.2f}, Silhouette score: {silhouette:.3f}. Distribusi: {cluster_summary}', 'success')
        else:
            db.session.rollback()
            flash('Tidak ada data cluster yang berhasil disimpan', 'error')
        
        return redirect(url_for('cluster.index'))
        
    except Exception as e:
        db.session.rollback()
        flash(f'Error dalam Penentuan Jumlah Cluster Optimal: {str(e)}', 'error')
        return redirect(url_for('cluster.index'))