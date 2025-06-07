from flask import Blueprint, render_template, request, redirect, url_for, flash
from app.models.KecamatanModel import Kecamatan
from app.forms.KecamatanForm import KecamatanForm
from app.extension import db

kecamatan_bp = Blueprint('kecamatan', __name__, url_prefix='/kecamatan')

@kecamatan_bp.route('/')
def index():
    data = Kecamatan.query.order_by(Kecamatan.nama_kecamatan.asc()).all()
    return render_template('kecamatan/index.html', data=data)

@kecamatan_bp.route('/tambah', methods=['GET', 'POST'])
@kecamatan_bp.route('/edit/<int:id>', methods=['GET', 'POST'])
def form(id=None):
    form = KecamatanForm()
    kecamatan = Kecamatan.query.get(id) if id else None

    if request.method == 'POST' and form.validate_on_submit():
        if kecamatan:
            kecamatan.nama_kecamatan = form.nama_kecamatan.data
            flash('Data berhasil diperbarui', 'success')
        else:
            kecamatan = Kecamatan(nama_kecamatan=form.nama_kecamatan.data)
            db.session.add(kecamatan)
            flash('Data berhasil ditambahkan', 'success')
        db.session.commit()
        return redirect(url_for('kecamatan.index'))

    if kecamatan:
        form.nama_kecamatan.data = kecamatan.nama_kecamatan

    return render_template('kecamatan/form.html', form=form, edit=bool(kecamatan))

@kecamatan_bp.route('/hapus/<int:id>')
def hapus(id):
    kecamatan = Kecamatan.query.get_or_404(id)
    db.session.delete(kecamatan)
    db.session.commit()
    flash('Data berhasil dihapus', 'info')
    return redirect(url_for('kecamatan.index'))
