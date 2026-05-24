# backend/src/infrastructure/bert_embedding_service.py
# BERT эмбеддинг сервис
#
# Версия: 1.1
# Обновлено: 2026-04-19
# Изменения: добавлен singleton паттерн для избежания повторной загрузки модели

"""
BERT эмбеддинг сервис для получения эмбеддингов через BERT модели.

Поддерживает:
- bert-base-multilingual-cased
- DeepPavlov/rubert-base-cased
- BERT модели для русского языка
"""

from typing import Optional
import numpy as np
import threading

from .embedding_service import EmbeddingService


# Глобальный кеш для singleton
_bert_instance: Optional["BertEmbeddingService"] = None
_bert_lock = threading.Lock()


def get_bert_embedding_service(model_name: str = "bert-base-multilingual-cased") -> "BertEmbeddingService":
    """Получить singleton экземпляр BertEmbeddingService.
    
    Модель BERT загружается только один раз и переиспользуется
    для всех запросов.
    
    Args:
        model_name: Название модели BERT.
    
    Returns:
        BertEmbeddingService: Singleton экземпляр.
    """
    global _bert_instance
    
    with _bert_lock:
        if _bert_instance is None or _bert_instance.model_name != model_name:
            _bert_instance = BertEmbeddingService(model_name=model_name)
        return _bert_instance


class BertEmbeddingService:
    """BERT эмбеддинг сервис.
    
    Использует BERT модели для получения эмбеддингов терминов.
    По умолчанию использует ruBERT модель.
    
    Singleton паттерн используется через get_bert_embedding_service().
    
    Attributes:
        model_name: Название модели HuggingFace
        embedding_dim: Размерность эмбеддингов
        _model: Модель transformers (lazy init)
        _tokenizer: Токенизатор (lazy init)
    """
    
    def __init__(
        self,
        model_name: str = "bert-base-multilingual-cased",
    ) -> None:
        """Инициализация.
        
        Args:
            model_name: Название модели BERT (по умолчанию multilingual)
        """
        self.model_name = model_name
        self._model = None
        self._tokenizer = None
        self._embedding_dim: Optional[int] = None
    
    def _ensure_model_loaded(self) -> None:
        """Ленивая загрузка модели и токенизатора."""
        if self._model is None:
            try:
                from transformers import AutoTokenizer, AutoModel
                
                self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                self._model = AutoModel.from_pretrained(self.model_name)
                self._model.eval()
                
                # Определяем размерность
                self._embedding_dim = self._model.config.hidden_size
                
            except ImportError:
                raise ImportError(
                    "transformers library required. Install: pip install transformers torch"
                )
    
    @property
    def embedding_dim(self) -> int:
        """Размерность эмбеддингов."""
        self._ensure_model_loaded()
        assert self._embedding_dim is not None
        return self._embedding_dim
    
    def get_embedding(self, text: str) -> np.ndarray:
        """Получить эмбеддинг для одного термина.
        
        Использует [CLS] токен для представления текста.
        
        Args:
            text: Текст для эмбеддинга
        
        Returns:
            Вектор эмбеддинга
        """
        self._ensure_model_loaded()
        
        if self._model is None or self._tokenizer is None:
            raise RuntimeError("Model not loaded")
        
        # Токенизация
        inputs = self._tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        )
        
        # Получаем эмбеддинги
        with np.testing.suppress_warnings():
            outputs = self._model(**inputs)
        
        # Используем [CLS] токен
        embedding = outputs.last_hidden_state[:, 0, :].detach().numpy()[0]
        
        return embedding
    
    def get_embeddings_batch(self, texts: list[str]) -> list[np.ndarray]:
        """Получить эмбеддинги для списка терминов.
        
        Args:
            texts: Список текстов для эмбеддингов
        
        Returns:
            Список векторов эмбеддингов
        """
        if not texts:
            return []
        
        self._ensure_model_loaded()
        
        if self._model is None or self._tokenizer is None:
            raise RuntimeError("Model not loaded")
        
        # Токенизация батчем
        inputs = self._tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        )
        
        # Получаем эмбеддинги
        with np.testing.suppress_warnings():
            outputs = self._model(**inputs)
        
        # [CLS] токены для всех текстов
        embeddings = outputs.last_hidden_state[:, 0, :].detach().numpy()
        
        return list(embeddings)
    
    def get_sentence_embedding(self, text: str) -> np.ndarray:
        """Получить sentence-level эмбеддинг через mean pooling.
        
        Альтернативный метод получения эмбеддинга - среднее по всем токенам,
        взвешенное по attention mask.
        
        Args:
            text: Текст
        
        Returns:
            Эмбеддинг предложения
        """
        self._ensure_model_loaded()
        
        if self._model is None or self._tokenizer is None:
            raise RuntimeError("Model not loaded")
        
        inputs = self._tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        )
        
        with np.testing.suppress_warnings():
            outputs = self._model(**inputs)
        
        # Mean pooling
        attention_mask = inputs["attention_mask"]
        hidden_states = outputs.last_hidden_state
        
        # Расширяем attention mask
        mask = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
        masked_hidden = hidden_states * mask
        
        # Сумма по токенам / сумма по маске
        summed = masked_hidden.sum(1)
        summed_mask = mask.sum(1)
        embedding = (summed / summed_mask).detach().numpy()[0]
        
        return embedding
