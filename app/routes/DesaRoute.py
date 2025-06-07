from flask import Blueprint, render_template, request, redirect, url_for, flash
from app.models.DesaModel import Desa
from app.models.KecamatanModel import Kecamatan
from app.forms.DesaForm import DesaForm
from app.extension import db

desa_bp = Blueprint('desa', __name__, url_prefix='/desa')

@desa_bp.route('/')
def index():
    data = Desa.query.order_by(Desa.nama_desa.asc()).all()
    return render_template('desa/index.html', data=data)

@desa_bp.route('/tambah', methods=['GET', 'POST'])
@desa_bp.route('/edit/<int:id_desa>', methods=['GET', 'POST'])
def form(id_desa=None):
    form = DesaForm()
    form.id_kecamatan.choices = [(k.id, k.nama_kecamatan) for k in Kecamatan.query.order_by(Kecamatan.nama_kecamatan).all()]

    desa = Desa.query.get(id_desa) if id_desa else None

    if request.method == 'POST' and form.validate_on_submit():
        if desa:
            desa.id_kecamatan = form.id_kecamatan.data
            desa.nama_desa = form.nama_desa.data
            flash('Data berhasil diperbarui', 'success')
        else:
            desa = Desa(id_kecamatan=form.id_kecamatan.data, nama_desa=form.nama_desa.data)
            db.session.add(desa)
            flash('Data berhasil ditambahkan', 'success')
        db.session.commit()
        return redirect(url_for('desa.index'))

    if desa:
        form.id_kecamatan.data = desa.id_kecamatan
        form.nama_desa.data = desa.nama_desa

    return render_template('desa/form.html', form=form, edit=bool(desa))

@desa_bp.route('/hapus/<int:id_desa>')
def hapus(id_desa):
    desa = Desa.query.get_or_404(id_desa)
    db.session.delete(desa)
    db.session.commit()
    flash('Data berhasil dihapus', 'info')
    return redirect(url_for('desa.index'))
