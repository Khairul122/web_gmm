from flask_wtf import FlaskForm
from wtforms import StringField, SubmitField
from wtforms.validators import DataRequired, Length

class KecamatanForm(FlaskForm):
    nama_kecamatan = StringField('Nama Kecamatan', validators=[DataRequired(), Length(3, 100)], render_kw={'placeholder': 'Masukkan nama kecamatan'})
    submit = SubmitField('Simpan')