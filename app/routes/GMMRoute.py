from flask import Blueprint, render_template, request, redirect, url_for, flash, jsonify
from app.models.GMMModel import GMM
from app.models.DesaModel import Desa
from app.models.ClusterModel import Cluster
from app.models.PerkebunanModel import Perkebunan as DataPerkebunan
from app.extension import db
import numpy as np
import json
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.metrics import davies_bouldin_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64

gmm_bp = Blueprint('gmm', __name__, url_prefix='/gmm')

@gmm_bp.route('/')
def index():
    data = GMM.query.join(Desa).join(Cluster).order_by(GMM.id_clustering.asc()).all()
    
    stats = {
        'total_clustering': len(data),
        'converged_count': len([item for item in data if item.converged]),
        'total_iterations': max([item.iteration for item in data], default=0),
        'clusters_count': len(set([item.id_cluster for item in data]))
    }
    
    return render_template('gmm/index.html', data=data, stats=stats)

@gmm_bp.route('/hapus/<int:id>')
def hapus(id):
    gmm_data = GMM.query.get_or_404(id)
    db.session.delete(gmm_data)
    db.session.commit()
    flash('Data GMM berhasil dihapus', 'info')
    return redirect(url_for('gmm.index'))

@gmm_bp.route('/hapus-semua')
def hapus_semua():
    GMM.query.delete()
    db.session.commit()
    flash('Semua data GMM berhasil dihapus', 'info')
    return redirect(url_for('gmm.index'))

@gmm_bp.route('/proses-gmm')
def proses_gmm():
    try:
        data_perkebunan = DataPerkebunan.query.all()
        
        if not data_perkebunan:
            flash('Tidak ada data perkebunan untuk diproses', 'error')
            return redirect(url_for('gmm.index'))
        
        clusters = Cluster.query.all()
        if not clusters:
            flash('Belum ada cluster yang terbentuk. Jalankan analisis BIC terlebih dahulu', 'error')
            return redirect(url_for('gmm.index'))
        
        GMM.query.delete()
        db.session.commit()
        
        features = []
        desa_ids = []
        for data in data_perkebunan:
            features.append([
                float(data.luas_tbm or 0),
                float(data.luas_tm or 0),
                float(data.luas_ttm or 0),
                float(data.produksi_ton or 0),
                int(data.jumlah_petani_kk or 0)
            ])
            desa_ids.append(data.id_desa)
        
        features = np.array(features)
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)
        
        n_clusters = len(set([cluster.nama_cluster for cluster in clusters]))
        
        gmm_model = GaussianMixture(
            n_components=n_clusters, 
            random_state=42, 
            max_iter=100,
            tol=1e-6
        )
        
        gmm_model.fit(features_scaled)
        
        cluster_labels = gmm_model.predict(features_scaled)
        probabilities = gmm_model.predict_proba(features_scaled)
        
        converged = gmm_model.converged_
        n_iter = gmm_model.n_iter_
        
        means = gmm_model.means_
        covariances = gmm_model.covariances_
        
        cluster_mapping = {}
        unique_cluster_names = list(set([cluster.nama_cluster for cluster in clusters]))
        for i, cluster_name in enumerate(unique_cluster_names):
            cluster_obj = Cluster.query.filter_by(nama_cluster=cluster_name).first()
            if cluster_obj:
                cluster_mapping[i] = cluster_obj.id_cluster
        
        for i, desa_id in enumerate(desa_ids):
            predicted_cluster_idx = cluster_labels[i]
            max_prob = np.max(probabilities[i])
            
            if predicted_cluster_idx in cluster_mapping:
                cluster_id = cluster_mapping[predicted_cluster_idx]
                
                mean_vector_json = json.dumps(means[predicted_cluster_idx].tolist())
                covariance_matrix_json = json.dumps(covariances[predicted_cluster_idx].tolist())
                
                gmm_data = GMM(
                    id_desa=desa_id,
                    id_cluster=cluster_id,
                    probabilitas=float(max_prob),
                    mean_vector=mean_vector_json,
                    covariance_matrix=covariance_matrix_json,
                    iteration=n_iter,
                    converged=converged
                )
                db.session.add(gmm_data)
        
        db.session.commit()
        
        flash(f'Proses GMM berhasil! {len(desa_ids)} desa telah diproses dengan {n_iter} iterasi. Konvergensi: {"Ya" if converged else "Tidak"}', 'success')
        return redirect(url_for('gmm.index'))
        
    except Exception as e:
        flash(f'Error dalam proses GMM: {str(e)}', 'error')
        return redirect(url_for('gmm.index'))

@gmm_bp.route('/detail/<int:id>')
def detail(id):
    gmm_data = GMM.query.get_or_404(id)
    
    try:
        mean_vector = json.loads(gmm_data.mean_vector) if gmm_data.mean_vector else []
        covariance_matrix = json.loads(gmm_data.covariance_matrix) if gmm_data.covariance_matrix else []
    except:
        mean_vector = []
        covariance_matrix = []
    
    detail_data = {
        'gmm': gmm_data,
        'mean_vector': mean_vector,
        'covariance_matrix': covariance_matrix,
        'feature_names': ['Luas TBM', 'Luas TM', 'Luas TTM', 'Produksi', 'Jumlah Petani']
    }
    
    return render_template('gmm/detail.html', data=detail_data)

@gmm_bp.route('/evaluasi-model')
def evaluasi_model():
    try:
        data_perkebunan = DataPerkebunan.query.all()
        gmm_data = GMM.query.all()
        
        if not data_perkebunan or not gmm_data:
            flash('Tidak ada data untuk evaluasi. Jalankan proses GMM terlebih dahulu', 'error')
            return redirect(url_for('gmm.index'))
        
        features = []
        cluster_labels = []
        for data in data_perkebunan:
            features.append([
                float(data.luas_tbm or 0),
                float(data.luas_tm or 0),
                float(data.luas_ttm or 0),
                float(data.produksi_ton or 0),
                int(data.jumlah_petani_kk or 0)
            ])
            
            gmm_item = next((g for g in gmm_data if g.id_desa == data.id_desa), None)
            if gmm_item:
                cluster_labels.append(gmm_item.id_cluster)
            else:
                cluster_labels.append(0)
        
        features = np.array(features)
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)
        
        silhouette_avg = silhouette_score(features_scaled, cluster_labels)
        davies_bouldin_idx = davies_bouldin_score(features_scaled, cluster_labels)
        
        plt.style.use('default')
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        scatter_plot = axes[0, 0]
        scatter = scatter_plot.scatter(features_scaled[:, 1], features_scaled[:, 3], 
                                      c=cluster_labels, cmap='viridis', alpha=0.7)
        scatter_plot.set_xlabel('Luas TM (Normalized)')
        scatter_plot.set_ylabel('Produksi (Normalized)')
        scatter_plot.set_title('Scatter Plot: Luas TM vs Produksi')
        plt.colorbar(scatter, ax=scatter_plot)
        
        heatmap_data = np.zeros((len(set(cluster_labels)), 5))
        for i, cluster_id in enumerate(set(cluster_labels)):
            cluster_features = features_scaled[np.array(cluster_labels) == cluster_id]
            if len(cluster_features) > 0:
                heatmap_data[i] = np.mean(cluster_features, axis=0)
        
        heatmap = axes[0, 1]
        sns.heatmap(heatmap_data, annot=True, fmt='.2f', cmap='RdYlGn', 
                   xticklabels=['TBM', 'TM', 'TTM', 'Produksi', 'Petani'],
                   yticklabels=[f'Cluster {i}' for i in range(len(set(cluster_labels)))],
                   ax=heatmap)
        heatmap.set_title('Heatmap Karakteristik Cluster')
        
        silhouette_plot = axes[1, 0]
        from sklearn.metrics import silhouette_samples
        silhouette_vals = silhouette_samples(features_scaled, cluster_labels)
        y_lower = 10
        for i, cluster_id in enumerate(set(cluster_labels)):
            cluster_silhouette_vals = silhouette_vals[np.array(cluster_labels) == cluster_id]
            cluster_silhouette_vals.sort()
            
            size_cluster_i = cluster_silhouette_vals.shape[0]
            y_upper = y_lower + size_cluster_i
            
            color = plt.cm.viridis(float(i) / len(set(cluster_labels)))
            silhouette_plot.fill_betweenx(np.arange(y_lower, y_upper),
                                        0, cluster_silhouette_vals,
                                        facecolor=color, edgecolor=color, alpha=0.7)
            y_lower = y_upper + 10
        
        silhouette_plot.axvline(x=silhouette_avg, color="red", linestyle="--", 
                               label=f'Rata-rata: {silhouette_avg:.3f}')
        silhouette_plot.set_xlabel('Silhouette Score')
        silhouette_plot.set_ylabel('Cluster')
        silhouette_plot.set_title('Silhouette Analysis')
        silhouette_plot.legend()
        
        evaluation_plot = axes[1, 1]
        metrics = ['Silhouette Score', 'Davies-Bouldin Index']
        values = [silhouette_avg, 1/davies_bouldin_idx]
        colors = ['green' if silhouette_avg > 0.5 else 'orange' if silhouette_avg > 0.3 else 'red',
                  'green' if davies_bouldin_idx < 1 else 'orange' if davies_bouldin_idx < 2 else 'red']
        
        bars = evaluation_plot.bar(metrics, values, color=colors, alpha=0.7)
        evaluation_plot.set_ylabel('Score')
        evaluation_plot.set_title('Evaluasi Kualitas Clustering')
        
        for bar, value in zip(bars, [silhouette_avg, davies_bouldin_idx]):
            height = bar.get_height()
            if 'Davies' in bar.get_x():
                evaluation_plot.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                                    f'{value:.3f}', ha='center', va='bottom')
            else:
                evaluation_plot.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                                    f'{value:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        
        img_buffer = io.BytesIO()
        plt.savefig(img_buffer, format='png', dpi=150, bbox_inches='tight')
        img_buffer.seek(0)
        plot_url = base64.b64encode(img_buffer.getvalue()).decode()
        plt.close()
        
        evaluation_data = {
            'silhouette_score': silhouette_avg,
            'davies_bouldin_index': davies_bouldin_idx,
            'plot_url': plot_url,
            'interpretation': {
                'silhouette': 'Baik' if silhouette_avg > 0.5 else 'Cukup' if silhouette_avg > 0.3 else 'Kurang',
                'davies_bouldin': 'Baik' if davies_bouldin_idx < 1 else 'Cukup' if davies_bouldin_idx < 2 else 'Kurang'
            }
        }
        
        return render_template('gmm/evaluasi.html', data=evaluation_data)
        
    except Exception as e:
        flash(f'Error dalam evaluasi model: {str(e)}', 'error')
        return redirect(url_for('gmm.index'))
    try:
        gmm_data = GMM.query.all()
        
        if not gmm_data:
            return jsonify({'error': 'Tidak ada data GMM'})
        
        cluster_stats = {}
        for item in gmm_data:
            cluster_name = item.cluster.nama_cluster
            if cluster_name not in cluster_stats:
                cluster_stats[cluster_name] = {
                    'count': 0,
                    'avg_probability': 0,
                    'total_probability': 0
                }
            
            cluster_stats[cluster_name]['count'] += 1
            cluster_stats[cluster_name]['total_probability'] += item.probabilitas or 0
        
        for cluster_name in cluster_stats:
            if cluster_stats[cluster_name]['count'] > 0:
                cluster_stats[cluster_name]['avg_probability'] = cluster_stats[cluster_name]['total_probability'] / cluster_stats[cluster_name]['count']
        
        return render_template('gmm/statistik.html', cluster_stats=cluster_stats)
        
    except Exception as e:
        flash(f'Error dalam menampilkan statistik: {str(e)}', 'error')
        return redirect(url_for('gmm.index'))