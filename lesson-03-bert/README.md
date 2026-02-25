
# Lesson 03 — BERT

## О чём занятие

Переход от статических эмбеддингов к контекстным.

Разбираем:

* self-attention
* Q, K, V
* softmax(QKᵀ / √d)
* padding mask и causal mask
* чем encoder отличается от decoder
* BERT

## Практика

* ручной расчёт attention на маленьком примере
* реализация attention в PyTorch
* извлечение BERT-эмбеддингов
* сравнение с Word2Vec
