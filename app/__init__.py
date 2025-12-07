from flask import Flask
from .model_loader import load_model

def create_app():
    """Создает и настраивает Flask приложение"""
    app = Flask(__name__)
    
    # Базовая конфигурация
    app.config['SECRET_KEY'] = 'plant_disease_detector_secret_key_2025'
    app.config['UPLOAD_FOLDER'] = 'app/static/uploads'
    app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
    app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}
    app.config['ENV'] = 'production'  # Явное указание окружения
    app.config['DEBUG'] = False  # Отключение debug режима в production
    
    # Создание папки для загрузок
    import os
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    
    # Загрузка модели при старте приложения
    print("🧠 Loading trained model...")
    try:
        app.model = load_model()
        print("✅ Model loaded successfully!")
    except Exception as e:
        print(f"❌ Error loading model: {str(e)}")
        app.model = None
    
    # Регистрация маршрутов
    from .routes import main_bp
    app.register_blueprint(main_bp)
    
    return app
