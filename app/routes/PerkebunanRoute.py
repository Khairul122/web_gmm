from flask import Blueprint, render_template, request, redirect, url_for, flash, jsonify
from app.models.PerkebunanModel import Perkebunan
from app.models.KecamatanModel import Kecamatan
from app.models.DesaModel import Desa
from app.forms.PerkebunanForm import PerkebunanForm
from app.extension import db
from sqlalchemy import or_

perkebunan_bp = Blueprint('perkebunan', __name__, url_prefix='/perkebunan')

@perkebunan_bp.route('/')
def index():
    try:
        page = request.args.get('page', 1, type=int)
        per_page = 20
        search = request.args.get('search', '')
        
        query = db.session.query(Perkebunan).join(
            Kecamatan, Perkebunan.id_kecamatan == Kecamatan.id
        ).join(
            Desa, Perkebunan.id_desa == Desa.id_desa
        )
        
        if search:
            query = query.filter(
                or_(
                    Kecamatan.nama_kecamatan.contains(search),
                    Desa.nama_desa.contains(search)
                )
            )
        
        pagination = query.order_by(Perkebunan.id_perkebunan.asc()).paginate(
            page=page, 
            per_page=per_page, 
            error_out=False
        )
        
        if pagination.total > 0 and page > pagination.pages:
            return redirect(url_for('perkebunan.index', page=1, search=search))
        
        return render_template('perkebunan/index.html', pagination=pagination)
    
    except Exception as e:
        flash(f'Error: {str(e)}', 'danger')
        return redirect(url_for('perkebunan.index'))

@perkebunan_bp.route('/tambah', methods=['GET', 'POST'])
@perkebunan_bp.route('/edit/<int:id_perkebunan>', methods=['GET', 'POST'])
def form(id_perkebunan=None):
    form = PerkebunanForm()
    
    try:
        form.id_kecamatan.choices = [(k.id, k.nama_kecamatan) for k in Kecamatan.query.order_by(Kecamatan.nama_kecamatan).all()]
        form.id_desa.choices = [(d.id_desa, d.nama_desa) for d in Desa.query.order_by(Desa.nama_desa).all()]

        perkebunan = Perkebunan.query.get(id_perkebunan) if id_perkebunan else None

        if request.method == 'POST' and form.validate_on_submit():
            if perkebunan:
                perkebunan.id_kecamatan = form.id_kecamatan.data
                perkebunan.id_desa = form.id_desa.data
                perkebunan.luas_tbm = form.luas_tbm.data
                perkebunan.luas_tm = form.luas_tm.data
                perkebunan.luas_ttm = form.luas_ttm.data
                perkebunan.luas_jumlah = form.luas_jumlah.data
                perkebunan.produksi_ton = form.produksi_ton.data
                perkebunan.produktivitas_kg_ha = form.produktivitas_kg_ha.data
                perkebunan.jumlah_petani_kk = form.jumlah_petani_kk.data
                flash('Data berhasil diperbarui', 'success')
            else:
                perkebunan = Perkebunan(
                    id_kecamatan=form.id_kecamatan.data,
                    id_desa=form.id_desa.data,
                    luas_tbm=form.luas_tbm.data,
                    luas_tm=form.luas_tm.data,
                    luas_ttm=form.luas_ttm.data,
                    luas_jumlah=form.luas_jumlah.data,
                    produksi_ton=form.produksi_ton.data,
                    produktivitas_kg_ha=form.produktivitas_kg_ha.data,
                    jumlah_petani_kk=form.jumlah_petani_kk.data
                )
                db.session.add(perkebunan)
                flash('Data berhasil ditambahkan', 'success')
            
            db.session.commit()
            return redirect(url_for('perkebunan.index'))

        if perkebunan:
            form.id_kecamatan.data = perkebunan.id_kecamatan
            form.id_desa.data = perkebunan.id_desa
            form.luas_tbm.data = perkebunan.luas_tbm
            form.luas_tm.data = perkebunan.luas_tm
            form.luas_ttm.data = perkebunan.luas_ttm
            form.luas_jumlah.data = perkebunan.luas_jumlah
            form.produksi_ton.data = perkebunan.produksi_ton
            form.produktivitas_kg_ha.data = perkebunan.produktivitas_kg_ha
            form.jumlah_petani_kk.data = perkebunan.jumlah_petani_kk

        return render_template('perkebunan/form.html', form=form, edit=bool(perkebunan))
    
    except Exception as e:
        flash(f'Error: {str(e)}', 'danger')
        return redirect(url_for('perkebunan.index'))

@perkebunan_bp.route('/hapus/<int:id_perkebunan>')
def hapus(id_perkebunan):
    try:
        perkebunan = Perkebunan.query.get_or_404(id_perkebunan)
        db.session.delete(perkebunan)
        db.session.commit()
        flash('Data berhasil dihapus', 'info')
    except Exception as e:
        flash(f'Error: {str(e)}', 'danger')
    
    return redirect(url_for('perkebunan.index'))

@perkebunan_bp.route('/desa/<int:id_kecamatan>')
def get_desa_by_kecamatan(id_kecamatan):
    try:
        desa_list = Desa.query.filter_by(id_kecamatan=id_kecamatan).order_by(Desa.nama_desa).all()
        result = [{'id': d.id_desa, 'nama': d.nama_desa} for d in desa_list]
        return jsonify({'desa': result})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@perkebunan_bp.route('/debug')
def debug():
    try:
        total_perkebunan = Perkebunan.query.count()
        total_kecamatan = Kecamatan.query.count()
        total_desa = Desa.query.count()
        
        sample_data = db.session.query(Perkebunan).join(
            Kecamatan, Perkebunan.id_kecamatan == Kecamatan.id
        ).join(
            Desa, Perkebunan.id_desa == Desa.id_desa
        ).limit(5).all()
        
        debug_info = {
            'total_perkebunan': total_perkebunan,
            'total_kecamatan': total_kecamatan,
            'total_desa': total_desa,
            'sample_data': []
        }
        
        for p in sample_data:
            debug_info['sample_data'].append({
                'id': p.id_perkebunan,
                'kecamatan': p.kecamatan.nama_kecamatan if p.kecamatan else 'N/A',
                'desa': p.desa.nama_desa if p.desa else 'N/A'
            })
        
        return jsonify(debug_info)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500