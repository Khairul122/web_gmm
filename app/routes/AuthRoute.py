from flask import Blueprint, render_template, request, redirect, url_for, flash, session
from app.models.AuthModel import User
from app.extension import db
from werkzeug.security import check_password_hash

auth_bp = Blueprint('auth', __name__, url_prefix='/auth')

@auth_bp.route('/')
def root():
    return redirect(url_for('auth.login'))

@auth_bp.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')

        user = User.query.filter_by(username=username).first()

        if user and (user.password == password or check_password_hash(user.password, password)):
            session['user_id'] = user.id_user
            session['username'] = user.username
            session['name'] = user.name
            session['level'] = user.level
            flash('Login berhasil', 'success')
            return redirect(url_for('dashboard.index')) 
        else:
            flash('Username atau password salah', 'danger')
            return redirect(url_for('auth.login'))

    return render_template('login/index.html')

@auth_bp.route('/logout')
def logout():
    session.clear()
    flash('Anda telah logout', 'info')
    return redirect(url_for('auth.login'))
