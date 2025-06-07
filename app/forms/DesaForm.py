from flask_wtf import FlaskForm
from wtforms import StringField, SubmitField, SelectField
from wtforms.validators import DataRequired, Length

class DesaForm(FlaskForm):
    id_kecamatan = SelectField('Kecamatan', coerce=int, validators=[DataRequired()])
    nama_desa = StringField('Nama Desa', validators=[DataRequired(), Length(3, 100)],
                            render_kw={'placeholder': 'Masukkan nama desa'})
    submit = SubmitField('Simpan')
