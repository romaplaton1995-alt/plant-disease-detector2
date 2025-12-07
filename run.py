from app import create_app
import torch

app = create_app()

@app.route('/health')
def health_check():
    """Эндпоинт для проверки здоровья приложения"""
    try:
        model_loaded = hasattr(app, 'model')
        gpu_available = torch.cuda.is_available() if model_loaded else False
        
        # Безопасное получение конфигурации с fallback значениями
        env = app.config.get('ENV', 'production')
        debug = app.config.get('DEBUG', False)
        
        return {
            'status': 'healthy',
            'model_loaded': model_loaded,
            'gpu_available': gpu_available,
            'environment': env,
            'debug_mode': debug,
            'app_name': 'Plant Disease Detector',
            'cuda_version': torch.version.cuda if gpu_available else None
        }
    except Exception as e:
        return {
            'status': 'unhealthy',
            'error': str(e),
            'environment': app.config.get('ENV', 'unknown')
        }, 500

if __name__ == '__main__':
    # Безопасное получение порта с fallback значением
    port = int(app.config.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
