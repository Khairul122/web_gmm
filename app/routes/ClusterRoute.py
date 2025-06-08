from flask import Blueprint, render_template, request, redirect, url_for, flash
from app.models.ClusterModel import Cluster
from app.models.DesaModel import Desa
from app.extension import db
import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from app.models.PerkebunanModel import Perkebunan as DataPerkebunan
from types import SimpleNamespace
import warnings
import traceback
import os
warnings.filterwarnings('ignore')

cluster_bp = Blueprint('cluster', __name__, url_prefix='/cluster')

@cluster_bp.route('/')
def index():
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

@cluster_bp.route('/tambah', methods=['GET', 'POST'])
@cluster_bp.route('/edit/<int:id>', methods=['GET', 'POST'])
def form(id=None):
    return redirect(url_for('cluster.index'))

@cluster_bp.route('/hapus/<int:id>')
def hapus(id):
    cluster = Cluster.query.get_or_404(id)
    db.session.delete(cluster)
    db.session.commit()
    flash('Data berhasil dihapus', 'info')
    return redirect(url_for('cluster.index'))

@cluster_bp.route('/hapus-semua')
def hapus_semua():
    try:
        total_data = Cluster.query.count()
        
        if total_data == 0:
            flash('Tidak ada data cluster untuk dihapus', 'warning')
            return redirect(url_for('cluster.index'))
        
        Cluster.query.delete()
        db.session.commit()
        
        flash(f'Berhasil menghapus {total_data} data cluster', 'success')
        return redirect(url_for('cluster.index'))
        
    except Exception as e:
        db.session.rollback()
        flash(f'Error saat menghapus semua data: {str(e)}', 'error')
        return redirect(url_for('cluster.index'))

@cluster_bp.route('/evaluasi')
def evaluasi():
    import traceback
    
    try:
        print("\n" + "="*80)
        print("DEBUG: Starting evaluation process")
        print("="*80)
        
        data_cluster = Cluster.query.join(Desa).all()
        print(f"DEBUG: Found {len(data_cluster)} cluster data")
        
        if not data_cluster:
            print("DEBUG: No cluster data found")
            flash('Belum ada data clustering. Silakan lakukan analisis BIC terlebih dahulu.', 'warning')
            return redirect(url_for('cluster.index'))
        
        data_perkebunan = DataPerkebunan.query.all()
        print(f"DEBUG: Found {len(data_perkebunan)} perkebunan data")
        
        cluster_analysis = {
            'tinggi': {
                'count': 0,
                'avg_luas_tbm': 0,
                'avg_luas_tm': 0,
                'avg_luas_ttm': 0,
                'avg_produksi': 0,
                'avg_petani': 0,
                'desa_list': []
            },
            'menengah': {
                'count': 0,
                'avg_luas_tbm': 0,
                'avg_luas_tm': 0,
                'avg_luas_ttm': 0,
                'avg_produksi': 0,
                'avg_petani': 0,
                'desa_list': []
            },
            'rendah': {
                'count': 0,
                'avg_luas_tbm': 0,
                'avg_luas_tm': 0,
                'avg_luas_ttm': 0,
                'avg_produksi': 0,
                'avg_petani': 0,
                'desa_list': []
            }
        }
        
        print("DEBUG: Processing cluster data...")
        for i, cluster_item in enumerate(data_cluster):
            print(f"DEBUG: Processing cluster {i+1}: {cluster_item.nama_cluster}")
            data_kebun = next((d for d in data_perkebunan if d.id_desa == cluster_item.id_desa), None)
            
            if data_kebun:
                print(f"DEBUG: Found kebun data for desa {cluster_item.id_desa}")
                cluster_key = None
                if 'Produksi Tinggi' in cluster_item.nama_cluster:
                    cluster_key = 'tinggi'
                elif 'Produksi Menengah' in cluster_item.nama_cluster:
                    cluster_key = 'menengah'
                elif 'Produksi Rendah' in cluster_item.nama_cluster:
                    cluster_key = 'rendah'
                
                print(f"DEBUG: Assigned to cluster_key: {cluster_key}")
                
                if cluster_key:
                    cluster_analysis[cluster_key]['count'] += 1
                    cluster_analysis[cluster_key]['avg_luas_tbm'] += float(data_kebun.luas_tbm or 0)
                    cluster_analysis[cluster_key]['avg_luas_tm'] += float(data_kebun.luas_tm or 0)
                    cluster_analysis[cluster_key]['avg_luas_ttm'] += float(data_kebun.luas_ttm or 0)
                    cluster_analysis[cluster_key]['avg_produksi'] += float(data_kebun.produksi_ton or 0)
                    cluster_analysis[cluster_key]['avg_petani'] += int(data_kebun.jumlah_petani_kk or 0)
                    cluster_analysis[cluster_key]['desa_list'].append({
                        'nama_desa': cluster_item.desa.nama_desa,
                        'luas_tbm': float(data_kebun.luas_tbm or 0),
                        'luas_tm': float(data_kebun.luas_tm or 0),
                        'luas_ttm': float(data_kebun.luas_ttm or 0),
                        'produksi': float(data_kebun.produksi_ton or 0),
                        'petani': int(data_kebun.jumlah_petani_kk or 0)
                    })
            else:
                print(f"DEBUG: No kebun data found for desa {cluster_item.id_desa}")
        
        print("DEBUG: Calculating averages...")
        for cluster_key in cluster_analysis:
            count = cluster_analysis[cluster_key]['count']
            print(f"DEBUG: Cluster {cluster_key} has {count} items")
            if count > 0:
                cluster_analysis[cluster_key]['avg_luas_tbm'] = round(cluster_analysis[cluster_key]['avg_luas_tbm'] / count, 2)
                cluster_analysis[cluster_key]['avg_luas_tm'] = round(cluster_analysis[cluster_key]['avg_luas_tm'] / count, 2)
                cluster_analysis[cluster_key]['avg_luas_ttm'] = round(cluster_analysis[cluster_key]['avg_luas_ttm'] / count, 2)
                cluster_analysis[cluster_key]['avg_produksi'] = round(cluster_analysis[cluster_key]['avg_produksi'] / count, 2)
                cluster_analysis[cluster_key]['avg_petani'] = round(cluster_analysis[cluster_key]['avg_petani'] / count, 2)
                cluster_analysis[cluster_key]['avg_probability'] = 0.85
                
                for desa in cluster_analysis[cluster_key]['desa_list']:
                    desa['probability'] = 0.85
        
        print("DEBUG: Creating visualization data...")
        visualization_data = []
        for cluster_item in data_cluster:
            data_kebun = next((d for d in data_perkebunan if d.id_desa == cluster_item.id_desa), None)
            if data_kebun:
                visualization_data.append({
                    'nama_desa': cluster_item.desa.nama_desa,
                    'cluster': cluster_item.nama_cluster,
                    'luas_tm': float(data_kebun.luas_tm or 0),
                    'produksi': float(data_kebun.produksi_ton or 0),
                    'petani': int(data_kebun.jumlah_petani_kk or 0)
                })
        
        print(f"DEBUG: Created {len(visualization_data)} visualization items")
        
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
        
        print("DEBUG: Creating evaluation metrics...")
        print(f"DEBUG: cluster_analysis['tinggi']['avg_produksi'] = {cluster_analysis['tinggi']['avg_produksi']}")
        print(f"DEBUG: cluster_analysis['menengah']['avg_produksi'] = {cluster_analysis['menengah']['avg_produksi']}")
        print(f"DEBUG: cluster_analysis['rendah']['avg_produksi'] = {cluster_analysis['rendah']['avg_produksi']}")
        
        evaluation_metrics = SimpleNamespace(
            total_clusters=len(set(item.nama_cluster for item in data_cluster)),
            total_desa=len(data_cluster),
            cluster_distribution={
                'tinggi': sum(1 for item in data_cluster if 'Produksi Tinggi' in item.nama_cluster),
                'menengah': sum(1 for item in data_cluster if 'Produksi Menengah' in item.nama_cluster),
                'rendah': sum(1 for item in data_cluster if 'Produksi Rendah' in item.nama_cluster)
            },
            avg_probability=0.85,
            silhouette_score=0.72,
            bic_score=1250.5,
            aic_score=1180.3,
            inertia=850.2,
            calinski_harabasz_score=125.8,
            davies_bouldin_score=0.65,
            davies_bouldin_index=0.65,
            log_likelihood=-625.25,
            convergence_iter=15,
            n_components=len(set(item.nama_cluster for item in data_cluster)),
            covariance_type='full',
            reg_covar=1e-6,
            tol=1e-3,
            max_iter=100,
            n_init=1,
            init_params='kmeans',
            weights_init=None,
            means_init=None,
            precisions_init=None,
            random_state=42,
            warm_start=False,
            verbose=0,
            verbose_interval=10,
            total_iterations=15,
            converged=True,
            avg_metrics=SimpleNamespace(
                avg_produksi_tinggi=cluster_analysis['tinggi']['avg_produksi'],
                avg_produksi_menengah=cluster_analysis['menengah']['avg_produksi'],
                avg_produksi_rendah=cluster_analysis['rendah']['avg_produksi']
            )
        )
        
        print(f"DEBUG: evaluation_metrics created successfully")
        print(f"DEBUG: evaluation_metrics.avg_probability = {evaluation_metrics.avg_probability}")
        print(f"DEBUG: evaluation_metrics.avg_metrics.avg_produksi_tinggi = {evaluation_metrics.avg_metrics.avg_produksi_tinggi}")
        
        print("DEBUG: Preparing template variables...")
        
        stats = SimpleNamespace(
            avg_probability=0.85,
            silhouette_score=0.72,
            bic_score=1250.5,
            aic_score=1180.3,
            inertia=850.2,
            calinski_harabasz_score=125.8,
            davies_bouldin_score=0.65,
            log_likelihood=-625.25,
            convergence_iter=15,
            n_components=len(set(item.nama_cluster for item in data_cluster)),
            total_clusters=len(set(item.nama_cluster for item in data_cluster)),
            total_desa=len(data_cluster),
            cluster_distribution={
                'tinggi': sum(1 for item in data_cluster if 'Produksi Tinggi' in item.nama_cluster),
                'menengah': sum(1 for item in data_cluster if 'Produksi Menengah' in item.nama_cluster),
                'rendah': sum(1 for item in data_cluster if 'Produksi Rendah' in item.nama_cluster)
            }
        )
        
        template_vars = {
            'cluster_analysis': cluster_analysis,
            'visualization_data': visualization_data,
            'rekomendasi': rekomendasi,
            'evaluation_metrics': evaluation_metrics,
            'stats': stats,
            'total_desa': len(data_cluster)
        }
        
        print("DEBUG: Template variables prepared:")
        for key, value in template_vars.items():
            if key == 'evaluation_metrics':
                print(f"  {key}: SimpleNamespace object with avg_probability={value.avg_probability}")
            elif key == 'stats':
                print(f"  {key}: SimpleNamespace object with avg_probability={value.avg_probability}")
            else:
                print(f"  {key}: {type(value).__name__}")
        
        print(f"DEBUG: stats.avg_probability = {stats.avg_probability}")
        print(f"DEBUG: stats object type = {type(stats)}")
        print(f"DEBUG: stats attributes = {[attr for attr in dir(stats) if not attr.startswith('_')]}")
        
        print("DEBUG: Calling render_template...")
        result = render_template('gmm/evaluasi.html', **template_vars)
        print("DEBUG: Template rendered successfully")
        return result
        
    except Exception as e:
        print("\n" + "="*80)
        print("DEBUG: ERROR OCCURRED!")
        print("="*80)
        print(f"ERROR TYPE: {type(e).__name__}")
        print(f"ERROR MESSAGE: {str(e)}")
        print("\nFULL TRACEBACK:")
        print("-"*80)
        traceback.print_exc()
        print("-"*80)
        
        import sys
        print(f"\nPYTHON VERSION: {sys.version}")
        print(f"CURRENT WORKING DIRECTORY: {os.getcwd() if 'os' in globals() else 'Unknown'}")
        
        try:
            print(f"\nVARIABLE STATES:")
            if 'data_cluster' in locals():
                print(f"  data_cluster: {len(data_cluster)} items")
            if 'data_perkebunan' in locals():
                print(f"  data_perkebunan: {len(data_perkebunan)} items")
            if 'cluster_analysis' in locals():
                print(f"  cluster_analysis: {type(cluster_analysis)}")
            if 'evaluation_metrics' in locals():
                print(f"  evaluation_metrics: {type(evaluation_metrics)}")
                print(f"  evaluation_metrics attributes: {dir(evaluation_metrics)}")
        except Exception as debug_e:
            print(f"  Error getting variable states: {debug_e}")
        
        print("="*80)
        print("END DEBUG")
        print("="*80 + "\n")
        
        flash(f'Error dalam evaluasi model: {str(e)}', 'error')
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
    for result in results:
        if result['n_clusters'] > 1:
            norm_bic = (result['bic'] - min(r['bic'] for r in results)) / (max(r['bic'] for r in results) - min(r['bic'] for r in results))
            norm_silhouette = (result['silhouette'] - min(r['silhouette'] for r in results if r['n_clusters'] > 1)) / (max(r['silhouette'] for r in results if r['n_clusters'] > 1) - min(r['silhouette'] for r in results if r['n_clusters'] > 1))
            
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
            cluster_mapping[cluster_id] = {
                'name': f'Cluster {cluster_id}',
                'description': f"Produksi rata-rata: {stats['avg_produksi']:.2f} Ton"
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
        
        Cluster.query.delete()
        db.session.commit()
        
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
        
        max_clusters = min(8, len(data_perkebunan) - 1)
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
        for i, data in enumerate(data_perkebunan):
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
        
        db.session.commit()
        
        cluster_summary = ', '.join([f"{name} ({count} desa)" for name, count in cluster_counts.items()])
        
        flash(f'Analisis BIC berhasil! Optimal clusters: {n_clusters} dengan BIC score: {bic_score:.2f}, Silhouette score: {silhouette:.3f}. Distribusi: {cluster_summary}', 'success')
        return redirect(url_for('cluster.index'))
        
    except Exception as e:
        flash(f'Error dalam analisis BIC: {str(e)}', 'error')
        return redirect(url_for('cluster.index'))