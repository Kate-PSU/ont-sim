// frontend/src/services/IndexedDBService.ts
// IndexedDB сервис для кеширования результатов бенчмарков
//
// Версия: 1.0
// Обновлено: 2026-04-09

/**
 * Типы для данных бенчмарка
 */
export interface MethodResult {
  method: string
  spearman: number
  pearson: number
  mse: number
  missing: number
  predictions_count: number
}

export interface BenchmarkComparison {
  dataset_name: string
  dataset_size: number
  execution_time_sec: number
  results: MethodResult[]
}

export interface StoredBenchmark {
  id?: number
  dataset: string
  comparison: BenchmarkComparison
  savedAt: string
  updatedAt: string
}

const DB_NAME = 'BenchmarkCacheDB'
const DB_VERSION = 1
const STORE_NAME = 'benchmarks'

class IndexedDBService {
  private db: IDBDatabase | null = null
  private dbReady: Promise<IDBDatabase> | null = null

  /**
   * Инициализация IndexedDB
   */
  async init(): Promise<IDBDatabase> {
    if (this.db) return this.db
    if (this.dbReady) return this.dbReady

    this.dbReady = new Promise((resolve, reject) => {
      const request = indexedDB.open(DB_NAME, DB_VERSION)

      request.onerror = () => {
        console.error('[IndexedDB] Ошибка открытия базы:', request.error)
        reject(request.error)
      }

      request.onsuccess = () => {
        this.db = request.result
        console.log('[IndexedDB] База данных открыта')
        resolve(this.db)
      }

      request.onupgradeneeded = (event) => {
        const db = (event.target as IDBOpenDBRequest).result
        
        // Создаём хранилище если не существует
        if (!db.objectStoreNames.contains(STORE_NAME)) {
          const store = db.createObjectStore(STORE_NAME, { keyPath: 'dataset' })
          store.createIndex('dataset', 'dataset', { unique: true })
          store.createIndex('savedAt', 'savedAt', { unique: false })
          console.log('[IndexedDB] Хранилище создано')
        }
      }
    })

    return this.dbReady
  }

  /**
   * Сохранение результатов бенчмарка
   */
  async saveBenchmark(dataset: string, comparison: BenchmarkComparison): Promise<void> {
    const db = await this.init()
    
    return new Promise((resolve, reject) => {
      const transaction = db.transaction([STORE_NAME], 'readwrite')
      const store = transaction.objectStore(STORE_NAME)
      
      const now = new Date().toISOString()
      const data: StoredBenchmark = {
        dataset,
        comparison,
        savedAt: now,
        updatedAt: now,
      }
      
      const request = store.put(data)
      
      request.onsuccess = () => {
        console.log(`[IndexedDB] Сохранено для ${dataset}`)
        resolve()
      }
      
      request.onerror = () => {
        console.error('[IndexedDB] Ошибка сохранения:', request.error)
        reject(request.error)
      }
    })
  }

  /**
   * Получение сохранённых результатов
   */
  async getBenchmark(dataset: string): Promise<BenchmarkComparison | null> {
    const db = await this.init()
    
    return new Promise((resolve, reject) => {
      const transaction = db.transaction([STORE_NAME], 'readonly')
      const store = transaction.objectStore(STORE_NAME)
      const request = store.get(dataset)
      
      request.onsuccess = () => {
        const result = request.result as StoredBenchmark | undefined
        if (result) {
          console.log(`[IndexedDB] Найдено для ${dataset}`)
          resolve(result.comparison)
        } else {
          console.log(`[IndexedDB] Не найдено для ${dataset}`)
          resolve(null)
        }
      }
      
      request.onerror = () => {
        console.error('[IndexedDB] Ошибка чтения:', request.error)
        reject(request.error)
      }
    })
  }

  /**
   * Получение всех сохранённых бенчмарков
   */
  async getAllBenchmarks(): Promise<StoredBenchmark[]> {
    const db = await this.init()
    
    return new Promise((resolve, reject) => {
      const transaction = db.transaction([STORE_NAME], 'readonly')
      const store = transaction.objectStore(STORE_NAME)
      const request = store.getAll()
      
      request.onsuccess = () => {
        resolve(request.result as StoredBenchmark[])
      }
      
      request.onerror = () => {
        console.error('[IndexedDB] Ошибка получения списка:', request.error)
        reject(request.error)
      }
    })
  }

  /**
   * Удаление результатов бенчмарка
   */
  async deleteBenchmark(dataset: string): Promise<void> {
    const db = await this.init()
    
    return new Promise((resolve, reject) => {
      const transaction = db.transaction([STORE_NAME], 'readwrite')
      const store = transaction.objectStore(STORE_NAME)
      const request = store.delete(dataset)
      
      request.onsuccess = () => {
        console.log(`[IndexedDB] Удалено для ${dataset}`)
        resolve()
      }
      
      request.onerror = () => {
        console.error('[IndexedDB] Ошибка удаления:', request.error)
        reject(request.error)
      }
    })
  }

  /**
   * Проверка наличия сохранённых результатов
   */
  async hasBenchmark(dataset: string): Promise<boolean> {
    const result = await this.getBenchmark(dataset)
    return result !== null
  }
}

// Экспорт singleton экземпляра
export const indexedDBService = new IndexedDBService()
export default indexedDBService