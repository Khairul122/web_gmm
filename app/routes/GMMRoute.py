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

        # Debug: Analisis distribusi data
        produksi_values = features[:, 3]
        print(f"Produksi - Min: {min(produksi_values):.2f}, Max: {max(produksi_values):.2f}, Mean: {np.mean(produksi_values):.2f}")
        
        # Debug: Cluster database yang tersedia
        print("Cluster database tersedia:")
        all_clusters = Cluster.query.all()
        for cluster in all_clusters:
            print(f"- {cluster.nama_cluster} (ID: {cluster.id_cluster})")

        # Force 3 clusters untuk konsistensi
        n_clusters = 3
        
        # Inisialisasi GMM dengan K-Means untuk stabilitas
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        kmeans_labels = kmeans.fit_predict(features_scaled)
        
        gmm_model = GaussianMixture(
            n_components=n_clusters,
            means_init=kmeans.cluster_centers_,
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

        # Hitung metrik evaluasi
        silhouette_avg = silhouette_score(features_scaled, cluster_labels)
        davies_bouldin = davies_bouldin_score(features_scaled, cluster_labels)
        
        print(f"GMM menghasilkan {len(np.unique(cluster_labels))} cluster unik")
        print(f"Cluster labels: {np.unique(cluster_labels)}")
        
        # Mapping cluster GMM ke cluster database - IMPROVED
        cluster_mapping = {}
        
        # Analisis karakteristik setiap cluster GMM
        cluster_characteristics = []
        for i in range(n_clusters):
            cluster_mask = cluster_labels == i
            cluster_features = features[cluster_mask]
            
            if len(cluster_features) > 0:
                avg_produksi = np.mean(cluster_features[:, 3])
                avg_luas_tm = np.mean(cluster_features[:, 1])
                avg_petani = np.mean(cluster_features[:, 4])
                
                cluster_characteristics.append({
                    'gmm_cluster_id': i,
                    'avg_produksi': avg_produksi,
                    'avg_luas_tm': avg_luas_tm,
                    'avg_petani': avg_petani,
                    'count': len(cluster_features)
                })
                
                print(f"Cluster GMM {i}: Produksi={avg_produksi:.2f}, Luas TM={avg_luas_tm:.2f}, Count={len(cluster_features)}")
        
        # Sort berdasarkan produksi (tinggi ke rendah)
        cluster_characteristics.sort(key=lambda x: x['avg_produksi'], reverse=True)
        
        # Ambil cluster database yang tersedia
        available_clusters = {
            'tinggi': Cluster.query.filter(Cluster.nama_cluster.like('%Tinggi%')).first(),
            'menengah': Cluster.query.filter(Cluster.nama_cluster.like('%Menengah%')).first(),
            'rendah': Cluster.query.filter(Cluster.nama_cluster.like('%Rendah%')).first()
        }
        
        print("Cluster database tersedia untuk mapping:")
        for level, cluster_obj in available_clusters.items():
            if cluster_obj:
                print(f"- {level}: {cluster_obj.nama_cluster} (ID: {cluster_obj.id_cluster})")
            else:
                print(f"- {level}: TIDAK DITEMUKAN")
        
        # Mapping berdasarkan ranking produksi
        mapping_order = ['tinggi', 'menengah', 'rendah']
        
        for idx, level in enumerate(mapping_order):
            if idx < len(cluster_characteristics) and available_clusters[level]:
                gmm_cluster_id = cluster_characteristics[idx]['gmm_cluster_id']
                cluster_mapping[gmm_cluster_id] = available_clusters[level].id_cluster
                print(f"Mapping: GMM Cluster {gmm_cluster_id} (Produksi: {cluster_characteristics[idx]['avg_produksi']:.2f}) -> {available_clusters[level].nama_cluster}")
        
        # Pastikan semua cluster GMM ter-mapping
        if len(cluster_mapping) != len(cluster_characteristics):
            print(f"WARNING: Hanya {len(cluster_mapping)} dari {len(cluster_characteristics)} cluster berhasil di-mapping!")
            
            # Mapping paksa untuk cluster yang belum ter-mapping
            used_cluster_ids = set(cluster_mapping.values())
            
            for char in cluster_characteristics:
                gmm_id = char['gmm_cluster_id']
                if gmm_id not in cluster_mapping:
                    # Cari cluster database yang belum digunakan
                    for level, cluster_obj in available_clusters.items():
                        if cluster_obj and cluster_obj.id_cluster not in used_cluster_ids:
                            cluster_mapping[gmm_id] = cluster_obj.id_cluster
                            used_cluster_ids.add(cluster_obj.id_cluster)
                            print(f"Mapping paksa: GMM Cluster {gmm_id} -> {cluster_obj.nama_cluster}")
                            break
        
        print(f"Final cluster mapping: {cluster_mapping}")
        
        # Simpan hasil GMM ke database
        gmm_results = []
        successful_assignments = 0
        
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
                    converged=converged,
                    silhouette_score=float(silhouette_avg),
                    davies_bouldin_index=float(davies_bouldin)
                )
                
                gmm_results.append(gmm_data)
                successful_assignments += 1
            else:
                print(f"ERROR: Desa {desa_id} tidak dapat di-assign ke cluster yang cocok")
        
        # Bulk insert untuk efisiensi
        if gmm_results:
            db.session.add_all(gmm_results)
            db.session.commit()
        
        # Pesan hasil
        if successful_assignments > 0:
            flash(f'Proses GMM berhasil! {successful_assignments} dari {len(desa_ids)} desa telah diproses dengan {n_iter} iterasi. '
                  f'Konvergensi: {"Ya" if converged else "Tidak"}. '
                  f'Silhouette Score: {silhouette_avg:.4f}, Davies-Bouldin Index: {davies_bouldin:.4f}', 'success')
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
        'feature_names': ['Luas TBM', 'Luas TM', 'Luas TTM', 'Produksi', 'Jumlah Petani']
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
            
            # Ambil metrik evaluasi dari record pertama
            first_record = gmm_data[0]
            evaluation_metrics = {
                'silhouette_score': first_record.silhouette_score or 0,
                'davies_bouldin_index': first_record.davies_bouldin_index or 0,
                'total_iterations': first_record.iteration or 0,
                'converged': first_record.converged or False
            }
            
            # Hitung statistik per cluster
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

            # Hitung rata-rata probabilitas
            for cluster_name in cluster_stats:
                if cluster_stats[cluster_name]['count'] > 0:
                    cluster_stats[cluster_name]['avg_probability'] = (
                        cluster_stats[cluster_name]['total_probability'] / 
                        cluster_stats[cluster_name]['count']
                    )
        
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
        
        # Analisis data per cluster
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
                        'desa_list': []
                    }
                
                cluster_analysis[cluster_name]['count'] += 1
                cluster_analysis[cluster_name]['avg_luas_tbm'] += float(data_kebun.luas_tbm or 0)
                cluster_analysis[cluster_name]['avg_luas_tm'] += float(data_kebun.luas_tm or 0)
                cluster_analysis[cluster_name]['avg_luas_ttm'] += float(data_kebun.luas_ttm or 0)
                cluster_analysis[cluster_name]['avg_produksi'] += float(data_kebun.produksi_ton or 0)
                cluster_analysis[cluster_name]['avg_petani'] += int(data_kebun.jumlah_petani_kk or 0)
                cluster_analysis[cluster_name]['total_probability'] += float(gmm_item.probabilitas or 0)
                
                cluster_analysis[cluster_name]['desa_list'].append({
                    'nama_desa': gmm_item.desa.nama_desa,
                    'luas_tbm': float(data_kebun.luas_tbm or 0),
                    'luas_tm': float(data_kebun.luas_tm or 0),
                    'luas_ttm': float(data_kebun.luas_ttm or 0),
                    'produksi': float(data_kebun.produksi_ton or 0),
                    'petani': int(data_kebun.jumlah_petani_kk or 0),
                    'probability': float(gmm_item.probabilitas or 0)
                })
                
                visualization_data.append({
                    'nama_desa': gmm_item.desa.nama_desa,
                    'cluster': cluster_name,
                    'luas_tm': float(data_kebun.luas_tm or 0),
                    'produksi': float(data_kebun.produksi_ton or 0),
                    'petani': int(data_kebun.jumlah_petani_kk or 0),
                    'probability': float(gmm_item.probabilitas or 0)
                })
        
        # Hitung rata-rata untuk setiap cluster
        for cluster_name in cluster_analysis:
            count = cluster_analysis[cluster_name]['count']
            if count > 0:
                cluster_analysis[cluster_name]['avg_luas_tbm'] = round(cluster_analysis[cluster_name]['avg_luas_tbm'] / count, 2)
                cluster_analysis[cluster_name]['avg_luas_tm'] = round(cluster_analysis[cluster_name]['avg_luas_tm'] / count, 2)
                cluster_analysis[cluster_name]['avg_luas_ttm'] = round(cluster_analysis[cluster_name]['avg_luas_ttm'] / count, 2)
                cluster_analysis[cluster_name]['avg_produksi'] = round(cluster_analysis[cluster_name]['avg_produksi'] / count, 2)
                cluster_analysis[cluster_name]['avg_petani'] = round(cluster_analysis[cluster_name]['avg_petani'] / count, 2)
                cluster_analysis[cluster_name]['avg_probability'] = round(cluster_analysis[cluster_name]['total_probability'] / count, 4)
        
        # Metrik evaluasi
        evaluation_metrics = {
            'silhouette_score': 0,
            'davies_bouldin_index': 0,
            'total_iterations': 0,
            'converged': False
        }
        
        if gmm_data:
            first_record = gmm_data[0]
            evaluation_metrics = {
                'silhouette_score': first_record.silhouette_score or 0,
                'davies_bouldin_index': first_record.davies_bouldin_index or 0,
                'total_iterations': first_record.iteration or 0,
                'converged': first_record.converged or False
            }
        
        # Rekomendasi berdasarkan hasil clustering
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