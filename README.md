# finodays
Папка text_classification содержит файлы для обучения ML моделей на датасете rureviews.

Для запуска django-приложения:

docker-compose -f local.yml build
docker-compose -f local.yml up

Доступные эндпоинты приложения:

ml/predict?endpoint_name=income_classifier

```
{
    "text": "Text"
}
```

Оценка текста по настроению