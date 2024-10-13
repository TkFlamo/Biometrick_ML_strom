# Biometrick_ML_strom

### Решение кейса 3

### Данные
Датасет lfw 
Результаты генерации на ArcFace эмбеддингах lfw 
Чекпоинты обученных моделей 
StyleGan2, ir_50 в папку pretrained_models
iteration_30000 в папку exp12/checkpoints (обученный маппер)
w600k_r50.onnx в корень (ArchFace - атакуемая модель)

### Ноутбуки
lfw_embed - создание эмбеддингов lfw и подсчет граничного значения сходства
mapper_test - тест обученной модели, подсчет attack success rate на lfw и тестовом датасете (1.0)
