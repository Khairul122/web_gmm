from flask_wtf import FlaskForm
from wtforms import DecimalField, IntegerField, SelectField, SubmitField
from wtforms.validators import DataRequired, NumberRange

class PerkebunanForm(FlaskForm):
    id_kecamatan = SelectField('Kecamatan', coerce=int, validators=[DataRequired()])
    id_desa = SelectField('Desa', coerce=int, validators=[DataRequired()])

    luas_tbm = DecimalField('Luas TBM (ha)', places=2, validators=[NumberRange(min=0)])
    luas_tm = DecimalField('Luas TM (ha)', places=2, validators=[NumberRange(min=0)])
    luas_ttm = DecimalField('Luas TTM (ha)', places=2, validators=[NumberRange(min=0)])
    luas_jumlah = DecimalField('Total Luas (ha)', places=2, validators=[NumberRange(min=0)])
    produksi_ton = DecimalField('Produksi (ton)', places=2, validators=[NumberRange(min=0)])
    produktivitas_kg_ha = DecimalField('Produktivitas (kg/ha)', places=2, validators=[NumberRange(min=0)])
    jumlah_petani_kk = IntegerField('Jumlah Petani (KK)', validators=[NumberRange(min=0)])

    submit = SubmitField('Simpan')
