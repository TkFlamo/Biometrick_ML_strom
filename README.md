# Biometrick_ML_strom

### Решение кейса 3

### Данные
Датасет lfw https://drive.google.com/file/d/1F3vM4SZtUqgB_WMSMo6O2-rc2ZCcC0U5/view?usp=sharing

Результаты генерации на ArcFace эмбеддингах lfw https://drive.google.com/file/d/1r0H7RVNGQCFnFuEFsgbbW71jbdeoUpW4/view?usp=sharing

Чекпоинты обученных моделей https://drive.google.com/file/d/1GuYiHSLXNQV5-8KXZO66OwrZtj6VhDrG/view?usp=sharing

StyleGan2, ir_50 в папку pretrained_models

iteration_30000 в папку exp12/checkpoints (обученный маппер)

w600k_r50.onnx в корень (ArchFace - атакуемая модель)

### Ноутбуки

lfw_embed - создание эмбеддингов lfw и подсчет граничного значения сходства

mapper_test - тест обученной модели, подсчет attack success rate на lfw и тестовом датасете (1.0)

### Результаты теста

attack success rate  100%

submit.zip - сгенерированные изображения
