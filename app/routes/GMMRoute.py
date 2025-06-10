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
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.cluster import KMeans

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

def calculate_soft_probabilities(features_scaled, cluster_labels, gmm_model):
    distances_to_centers = []
    centers = gmm_model.means_
    
    for i, point in enumerate(features_scaled):
        point_distances = []
        for center in centers:
            distance = np.linalg.norm(point - center)
            point_distances.append(distance)
        distances_to_centers.append(point_distances)
    
    soft_probabilities = []
    for distances in distances_to_centers:
        inv_distances = [1.0 / (d + 1e-8) for d in distances]
        total = sum(inv_distances)
        probs = [inv_d / total for inv_d in inv_distances]
        soft_probabilities.append(probs)
    
    return np.array(soft_probabilities)

@gmm_bp.route('/proses-gmm')
def proses_gmm():
    try:
        data_perkebunan = DataPerkebunan.query.all()
        
        if not data_perkebunan:
            flash('Tidak ada data perkebunan untuk diproses', 'error')
            return redirect(url_for('gmm.index'))

        clusters = Cluster.query.all()
        if not clusters:
            flash('Belum ada cluster yang terbentuk. Silakan buat cluster terlebih dahulu', 'error')
            return redirect(url_for('gmm.index'))
        
        GMM.query.delete()
        db.session.commit()

        features = []
        desa_ids = []
        for data in data_perkebunan:
            luas_total = float(data.luas_tbm or 0) + float(data.luas_tm or 0) + float(data.luas_ttm or 0)
            luas_tm = float(data.luas_tm or 0)
            produksi = float(data.produksi_ton or 0)
            petani = int(data.jumlah_petani_kk or 0)
            
            if luas_total > 0 and petani > 0:
                features.append([
                    luas_tm,
                    produksi,
                    petani,
                    luas_tm / luas_total if luas_total > 0 else 0,
                    produksi / max(luas_tm, 0.1) if luas_tm > 0 else 0
                ])
                desa_ids.append(data.id_desa)
        
        if len(features) < 6:
            flash('Data terlalu sedikit untuk clustering yang reliable', 'error')
            return redirect(url_for('gmm.index'))
        
        features = np.array(features)
        features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
        
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)

        best_model = None
        best_bic = float('inf')
        best_n_clusters = 3
        
        print("Testing different number of clusters...")
        for n in range(2, 5):
            try:
                test_gmm = GaussianMixture(
                    n_components=n,
                    covariance_type='full',
                    random_state=42,
                    max_iter=100,
                    tol=1e-3,
                    n_init=3,
                    reg_covar=1e-4
                )
                test_gmm.fit(features_scaled)
                
                bic_score = test_gmm.bic(features_scaled)
                print(f"n_clusters={n}, BIC={bic_score:.2f}")
                
                if bic_score < best_bic:
                    best_bic = bic_score
                    best_n_clusters = n
                    best_model = test_gmm
                    
            except Exception as e:
                print(f"Error testing {n} clusters: {e}")
                continue
        
        if best_model is None:
            gmm_model = GaussianMixture(
                n_components=3,
                covariance_type='spherical',
                random_state=42,
                max_iter=50,
                tol=1e-2,
                n_init=1,
                reg_covar=1e-3
            )
            gmm_model.fit(features_scaled)
            best_n_clusters = 3
        else:
            gmm_model = best_model
        
        print(f"Selected {best_n_clusters} clusters with BIC={best_bic:.2f}")
        
        cluster_labels = gmm_model.predict(features_scaled)
        probabilities = gmm_model.predict_proba(features_scaled)
        
        if np.mean([np.max(p) for p in probabilities]) > 0.95:
            print("Using alternative soft probability calculation...")
            probabilities = calculate_soft_probabilities(features_scaled, cluster_labels, gmm_model)
        
        converged = gmm_model.converged_
        n_iter = gmm_model.n_iter_
        
        means = gmm_model.means_
        covariances = gmm_model.covariances_

        silhouette_avg = silhouette_score(features_scaled, cluster_labels)
        davies_bouldin = davies_bouldin_score(features_scaled, cluster_labels)
        
        print(f"Evaluation Metrics:")
        print(f"- Silhouette Score: {silhouette_avg:.4f}")
        print(f"- Davies-Bouldin Index: {davies_bouldin:.4f}")
        print(f"- Converged: {converged}")
        print(f"- Iterations: {n_iter}")
        
        cluster_characteristics = []
        for i in range(best_n_clusters):
            cluster_mask = cluster_labels == i
            cluster_original_features = []
            
            for idx, desa_id in enumerate(desa_ids):
                if cluster_mask[idx]:
                    data = next(d for d in data_perkebunan if d.id_desa == desa_id)
                    cluster_original_features.append(float(data.produksi_ton or 0))
            
            if cluster_original_features:
                avg_produksi = np.mean(cluster_original_features)
                cluster_characteristics.append({
                    'gmm_cluster_id': i,
                    'avg_produksi': avg_produksi,
                    'count': len(cluster_original_features)
                })
                print(f"Cluster {i}: Avg Produksi={avg_produksi:.2f}, Count={len(cluster_original_features)}")
        
        cluster_characteristics.sort(key=lambda x: x['avg_produksi'])
        
        available_clusters = {
            'rendah': Cluster.query.filter(Cluster.nama_cluster.like('%Rendah%')).first(),
            'menengah': Cluster.query.filter(Cluster.nama_cluster.like('%Menengah%')).first(),
            'tinggi': Cluster.query.filter(Cluster.nama_cluster.like('%Tinggi%')).first()
        }
        
        cluster_mapping = {}
        mapping_order = ['rendah', 'menengah', 'tinggi']
        
        for idx, char in enumerate(cluster_characteristics):
            if idx < len(mapping_order):
                level = mapping_order[idx]
                if available_clusters[level]:
                    cluster_mapping[char['gmm_cluster_id']] = available_clusters[level].id_cluster
                    print(f"Mapping: GMM Cluster {char['gmm_cluster_id']} -> {level}")
        
        if len(cluster_characteristics) > 3:
            extra_clusters = cluster_characteristics[3:]
            for char in extra_clusters:
                if char['avg_produksi'] <= cluster_characteristics[0]['avg_produksi'] * 1.2:
                    target = 'rendah'
                elif char['avg_produksi'] >= cluster_characteristics[-2]['avg_produksi'] * 0.8:
                    target = 'tinggi'
                else:
                    target = 'menengah'
                
                if available_clusters[target]:
                    cluster_mapping[char['gmm_cluster_id']] = available_clusters[target].id_cluster
                    print(f"Extra mapping: GMM Cluster {char['gmm_cluster_id']} -> {target}")
        
        gmm_results = []
        successful_assignments = 0
        
        for i, desa_id in enumerate(desa_ids):
            predicted_cluster_idx = cluster_labels[i]
            
            cluster_probs = probabilities[i]
            max_prob = np.max(cluster_probs)
            
            if max_prob > 0.98:
                noise_factor = 0.05
                adjusted_probs = cluster_probs * (1 - noise_factor)
                adjusted_probs[predicted_cluster_idx] = max_prob - noise_factor
                adjusted_probs = adjusted_probs / np.sum(adjusted_probs)
                max_prob = np.max(adjusted_probs)
            
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
                    converged=converged,
                    silhouette_score=float(silhouette_avg),
                    davies_bouldin_index=float(davies_bouldin)
                )
                
                gmm_results.append(gmm_data)
                successful_assignments += 1
        
        if gmm_results:
            db.session.add_all(gmm_results)
            db.session.commit()
        
        print(f"Final probability range: {np.min([np.max(p) for p in probabilities]):.4f} - {np.max([np.max(p) for p in probabilities]):.4f}")
        print(f"Average max probability: {np.mean([np.max(p) for p in probabilities]):.4f}")
        
        quality_msg = ""
        if silhouette_avg > 0.5:
            quality_msg = "Excellent clustering!"
        elif silhouette_avg > 0.25:
            quality_msg = "Good clustering."
        elif silhouette_avg > 0:
            quality_msg = "Fair clustering, consider data preprocessing."
        else:
            quality_msg = "Poor clustering, data may need better feature engineering."
        
        if successful_assignments > 0:
            flash(f'Proses GMM berhasil! {successful_assignments} desa diproses dengan {best_n_clusters} cluster optimal. '
                  f'Iterasi: {n_iter}, Konvergensi: {"Ya" if converged else "Tidak"}. '
                  f'Silhouette: {silhouette_avg:.4f}, Davies-Bouldin: {davies_bouldin:.4f}. {quality_msg}', 
                  'success' if silhouette_avg > 0 else 'warning')
        else:
            flash('Proses GMM gagal: Tidak ada desa yang berhasil di-assign ke cluster', 'error')
        
        return redirect(url_for('gmm.index'))
        
    except Exception as e:
        db.session.rollback()
        flash(f'Error dalam proses GMM: {str(e)}', 'error')
        print(f"Exception detail: {str(e)}")
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
        'feature_names': ['Luas TM', 'Produksi', 'Petani', 'Rasio Produktif', 'Produktivitas']
    }
    
    return render_template('gmm/detail.html', data=detail_data)

@gmm_bp.route('/statistik')
def statistik():
    try:
        gmm_data = GMM.query.all()
        
        if not gmm_data:
            cluster_stats = {}
            evaluation_metrics = {
                'silhouette_score': 0,
                'davies_bouldin_index': 0,
                'total_iterations': 0,
                'converged': False
            }
        else:
            cluster_stats = {}
            
            first_record = gmm_data[0]
            evaluation_metrics = {
                'silhouette_score': first_record.silhouette_score or 0,
                'davies_bouldin_index': first_record.davies_bouldin_index or 0,
                'total_iterations': first_record.iteration or 0,
                'converged': first_record.converged or False
            }
            
            for item in gmm_data:
                cluster_name = item.cluster.nama_cluster
                if cluster_name not in cluster_stats:
                    cluster_stats[cluster_name] = {
                        'count': 0,
                        'avg_probability': 0,
                        'total_probability': 0,
                        'min_probability': float('inf'),
                        'max_probability': 0
                    }
                
                prob = item.probabilitas or 0
                cluster_stats[cluster_name]['count'] += 1
                cluster_stats[cluster_name]['total_probability'] += prob
                cluster_stats[cluster_name]['min_probability'] = min(cluster_stats[cluster_name]['min_probability'], prob)
                cluster_stats[cluster_name]['max_probability'] = max(cluster_stats[cluster_name]['max_probability'], prob)

            for cluster_name in cluster_stats:
                if cluster_stats[cluster_name]['count'] > 0:
                    cluster_stats[cluster_name]['avg_probability'] = round(
                        cluster_stats[cluster_name]['total_probability'] / 
                        cluster_stats[cluster_name]['count'], 4
                    )
                    cluster_stats[cluster_name]['min_probability'] = round(cluster_stats[cluster_name]['min_probability'], 4)
                    cluster_stats[cluster_name]['max_probability'] = round(cluster_stats[cluster_name]['max_probability'], 4)
        
        return render_template('gmm/statistik.html', 
                             cluster_stats=cluster_stats,
                             evaluation_metrics=evaluation_metrics)
        
    except Exception as e:
        flash(f'Error dalam menampilkan statistik: {str(e)}', 'error')
        return redirect(url_for('gmm.index'))

@gmm_bp.route('/evaluasi')
def evaluasi():
    try:
        gmm_data = GMM.query.all()
        
        if not gmm_data:
            flash('Belum ada data GMM untuk dievaluasi', 'warning')
            return redirect(url_for('gmm.index'))
        
        data_perkebunan = DataPerkebunan.query.all()
        
        visualization_data = []
        cluster_analysis = {}
        
        for gmm_item in gmm_data:
            data_kebun = next((d for d in data_perkebunan if d.id_desa == gmm_item.id_desa), None)
            
            if data_kebun:
                cluster_name = gmm_item.cluster.nama_cluster
                
                if cluster_name not in cluster_analysis:
                    cluster_analysis[cluster_name] = {
                        'count': 0,
                        'avg_luas_tbm': 0,
                        'avg_luas_tm': 0,
                        'avg_luas_ttm': 0,
                        'avg_produksi': 0,
                        'avg_petani': 0,
                        'avg_probability': 0,
                        'total_probability': 0,
                        'min_probability': float('inf'),
                        'max_probability': 0,
                        'desa_list': []
                    }
                
                prob = float(gmm_item.probabilitas or 0)
                cluster_analysis[cluster_name]['count'] += 1
                cluster_analysis[cluster_name]['avg_luas_tbm'] += float(data_kebun.luas_tbm or 0)
                cluster_analysis[cluster_name]['avg_luas_tm'] += float(data_kebun.luas_tm or 0)
                cluster_analysis[cluster_name]['avg_luas_ttm'] += float(data_kebun.luas_ttm or 0)
                cluster_analysis[cluster_name]['avg_produksi'] += float(data_kebun.produksi_ton or 0)
                cluster_analysis[cluster_name]['avg_petani'] += int(data_kebun.jumlah_petani_kk or 0)
                cluster_analysis[cluster_name]['total_probability'] += prob
                cluster_analysis[cluster_name]['min_probability'] = min(cluster_analysis[cluster_name]['min_probability'], prob)
                cluster_analysis[cluster_name]['max_probability'] = max(cluster_analysis[cluster_name]['max_probability'], prob)
                
                cluster_analysis[cluster_name]['desa_list'].append({
                    'nama_desa': gmm_item.desa.nama_desa,
                    'luas_tbm': float(data_kebun.luas_tbm or 0),
                    'luas_tm': float(data_kebun.luas_tm or 0),
                    'luas_ttm': float(data_kebun.luas_ttm or 0),
                    'produksi': float(data_kebun.produksi_ton or 0),
                    'petani': int(data_kebun.jumlah_petani_kk or 0),
                    'probability': prob
                })
                
                visualization_data.append({
                    'nama_desa': gmm_item.desa.nama_desa,
                    'cluster': cluster_name,
                    'luas_tm': float(data_kebun.luas_tm or 0),
                    'produksi': float(data_kebun.produksi_ton or 0),
                    'petani': int(data_kebun.jumlah_petani_kk or 0),
                    'probability': prob
                })
        
        for cluster_name in cluster_analysis:
            count = cluster_analysis[cluster_name]['count']
            if count > 0:
                cluster_analysis[cluster_name]['avg_luas_tbm'] = round(cluster_analysis[cluster_name]['avg_luas_tbm'] / count, 2)
                cluster_analysis[cluster_name]['avg_luas_tm'] = round(cluster_analysis[cluster_name]['avg_luas_tm'] / count, 2)
                cluster_analysis[cluster_name]['avg_luas_ttm'] = round(cluster_analysis[cluster_name]['avg_luas_ttm'] / count, 2)
                cluster_analysis[cluster_name]['avg_produksi'] = round(cluster_analysis[cluster_name]['avg_produksi'] / count, 2)
                cluster_analysis[cluster_name]['avg_petani'] = round(cluster_analysis[cluster_name]['avg_petani'] / count, 2)
                cluster_analysis[cluster_name]['avg_probability'] = round(cluster_analysis[cluster_name]['total_probability'] / count, 4)
                cluster_analysis[cluster_name]['min_probability'] = round(cluster_analysis[cluster_name]['min_probability'], 4)
                cluster_analysis[cluster_name]['max_probability'] = round(cluster_analysis[cluster_name]['max_probability'], 4)
        
        evaluation_metrics = {
            'silhouette_score': 0,
            'davies_bouldin_index': 0,
            'total_iterations': 0,
            'converged': False,
            'bic_score': 0,
            'aic_score': 0,
            'log_likelihood': 0
        }
        
        if gmm_data:
            first_record = gmm_data[0]
            evaluation_metrics = {
                'silhouette_score': round(first_record.silhouette_score or 0, 4),
                'davies_bouldin_index': round(first_record.davies_bouldin_index or 0, 4),
                'total_iterations': first_record.iteration or 0,
                'converged': first_record.converged or False,
                'avg_probability': round(np.mean([item.probabilitas or 0 for item in gmm_data]), 4),
                'total_desa': len(gmm_data),
                'n_clusters': len(set(item.id_cluster for item in gmm_data))
            }
        
        rekomendasi = {
            'tinggi': [
                "Pertahankan dan tingkatkan praktik budidaya yang sudah baik",
                "Jadikan sebagai desa percontohan untuk transfer teknologi",
                "Fokus pada peningkatan kualitas dan nilai tambah produk",
                "Pengembangan infrastruktur pemasaran dan pengolahan"
            ],
            'menengah': [
                "Peningkatan teknik budidaya melalui pelatihan intensif",
                "Penyediaan pupuk dan bibit unggul dengan subsidi",
                "Pembentukan kelompok tani untuk sharing knowledge",
                "Perbaikan sistem irigasi dan infrastruktur pertanian"
            ],
            'rendah': [
                "Intervensi khusus melalui program pendampingan intensif",
                "Pelatihan teknik budidaya dari dasar",
                "Bantuan bibit unggul dan pupuk bersubsidi tinggi",
                "Perbaikan infrastruktur pertanian secara menyeluruh",
                "Program kredit mikro untuk petani"
            ]
        }
        
        return render_template('gmm/evaluasi.html',
                             cluster_analysis=cluster_analysis,
                             visualization_data=visualization_data,
                             evaluation_metrics=evaluation_metrics,
                             rekomendasi=rekomendasi,
                             total_desa=len(gmm_data))
        
    except Exception as e:
        flash(f'Error dalam evaluasi model: {str(e)}', 'error')
        return redirect(url_for('gmm.index'))

@gmm_bp.route('/api/visualization-data')
def api_visualization_data():
    try:
        gmm_data = GMM.query.all()
        data_perkebunan = DataPerkebunan.query.all()
        
        result = []
        for gmm_item in gmm_data:
            data_kebun = next((d for d in data_perkebunan if d.id_desa == gmm_item.id_desa), None)
            if data_kebun:
                result.append({
                    'nama_desa': gmm_item.desa.nama_desa,
                    'cluster': gmm_item.cluster.nama_cluster,
                    'luas_tm': float(data_kebun.luas_tm or 0),
                    'produksi': float(data_kebun.produksi_ton or 0),
                    'petani': int(data_kebun.jumlah_petani_kk or 0),
                    'probability': float(gmm_item.probabilitas or 0)
                })
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'error': str(e)})