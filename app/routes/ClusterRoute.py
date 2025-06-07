from flask import Blueprint, render_template, request, redirect, url_for, flash
from app.models.ClusterModel import Cluster
from app.models.DesaModel import Desa
from app.extension import db
import numpy as np
from app.models.PerkebunanModel import Perkebunan as DataPerkebunan

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

@cluster_bp.route('/evaluasi')
def evaluasi():
    """
    Route untuk menampilkan halaman evaluasi model clustering
    """
    try:
        data_cluster = Cluster.query.join(Desa).all()
        
        if not data_cluster:
            flash('Belum ada data clustering. Silakan lakukan analisis BIC terlebih dahulu.', 'warning')
            return redirect(url_for('cluster.index'))
        
        data_perkebunan = DataPerkebunan.query.all()
        
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
        
        for cluster_item in data_cluster:
            data_kebun = next((d for d in data_perkebunan if d.id_desa == cluster_item.id_desa), None)
            
            if data_kebun:
                cluster_key = None
                if 'Produksi Tinggi' in cluster_item.nama_cluster:
                    cluster_key = 'tinggi'
                elif 'Produksi Menengah' in cluster_item.nama_cluster:
                    cluster_key = 'menengah'
                elif 'Produksi Rendah' in cluster_item.nama_cluster:
                    cluster_key = 'rendah'
                
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
        
        for cluster_key in cluster_analysis:
            if cluster_analysis[cluster_key]['count'] > 0:
                count = cluster_analysis[cluster_key]['count']
                cluster_analysis[cluster_key]['avg_luas_tbm'] = round(cluster_analysis[cluster_key]['avg_luas_tbm'] / count, 2)
                cluster_analysis[cluster_key]['avg_luas_tm'] = round(cluster_analysis[cluster_key]['avg_luas_tm'] / count, 2)
                cluster_analysis[cluster_key]['avg_luas_ttm'] = round(cluster_analysis[cluster_key]['avg_luas_ttm'] / count, 2)
                cluster_analysis[cluster_key]['avg_produksi'] = round(cluster_analysis[cluster_key]['avg_produksi'] / count, 2)
                cluster_analysis[cluster_key]['avg_petani'] = round(cluster_analysis[cluster_key]['avg_petani'] / count, 2)
        
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
                             rekomendasi=rekomendasi,
                             total_desa=len(data_cluster))
        
    except Exception as e:
        flash(f'Error dalam evaluasi model: {str(e)}', 'error')
        return redirect(url_for('cluster.index'))

@cluster_bp.route('/analisis-bic')
def analisis_bic():
    try:   
        data_perkebunan = DataPerkebunan.query.all()
        
        if not data_perkebunan:
            flash('Tidak ada data perkebunan untuk dianalisis', 'error')
            return redirect(url_for('cluster.index'))
        
        Cluster.query.delete()
        db.session.commit()
        
        cluster_0_count = 0
        cluster_1_count = 0
        cluster_2_count = 0
        
        for data in data_perkebunan:
            luas_tbm = float(data.luas_tbm or 0)
            luas_tm = float(data.luas_tm or 0)
            luas_ttm = float(data.luas_ttm or 0)
            produksi_ton = float(data.produksi_ton or 0)
            jumlah_petani_kk = int(data.jumlah_petani_kk or 0)
            
            cluster_name = ""
            cluster_desc = ""
            
            cluster_0_score = abs(luas_tbm - 107) + abs(luas_tm - 25) + abs(luas_ttm - 6) + abs(produksi_ton - 31) + abs(jumlah_petani_kk - 50)
            cluster_1_score = abs(luas_tbm - 12) + abs(luas_tm - 16.25) + abs(produksi_ton - 15) + abs(jumlah_petani_kk - 33)
            cluster_2_score = abs(luas_tbm - 106.33) + abs(luas_tm - 33.17) + abs(produksi_ton - 31) + abs(jumlah_petani_kk - 50)
            
            min_score = min(cluster_0_score, cluster_1_score, cluster_2_score)
            
            if min_score == cluster_0_score:
                cluster_name = "Produksi Menengah"
                cluster_desc = "Luas TBM rata-rata: 107 Ha, Luas TM rata-rata: 25 Ha, Luas TTM rata-rata: 6 Ha, Produksi rata-rata: 31 Ton, Jumlah petani: ±50 KK"
                cluster_0_count += 1
            elif min_score == cluster_1_score:
                cluster_name = "Produksi Rendah"
                cluster_desc = "Luas TBM rata-rata: 12 Ha, Luas TM rata-rata: 16,25 Ha, Produksi rata-rata: 15 Ton, Jumlah petani: ±33 KK. Menggambarkan wilayah dengan potensi rendah dan perlu intervensi khusus (pelatihan, bibit unggul)."
                cluster_1_count += 1
            else:
                cluster_name = "Produksi Tinggi"
                cluster_desc = "Luas TBM rata-rata: 106,33 Ha, Luas TM rata-rata: 33,17 Ha, Produksi rata-rata: 31 Ton, Jumlah petani: ±50 KK"
                cluster_2_count += 1
            
            cluster = Cluster(
                id_desa=data.id_desa,
                nama_cluster=cluster_name,
                deskripsi=cluster_desc
            )
            db.session.add(cluster)
        
        db.session.commit()
        
        total_assigned = cluster_0_count + cluster_1_count + cluster_2_count
        flash(f'Analisis BIC berhasil! {total_assigned} desa telah dikelompokkan: Produksi Menengah ({cluster_0_count} desa), Produksi Rendah ({cluster_1_count} desa), Produksi Tinggi ({cluster_2_count} desa)', 'success')
        return redirect(url_for('cluster.index'))
        
    except Exception as e:
        flash(f'Error dalam analisis BIC: {str(e)}', 'error')
        return redirect(url_for('cluster.index'))